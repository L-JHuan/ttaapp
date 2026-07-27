"""从统一协议的事件与 POI 表构造 GNPR 的 79 维连续表示。"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from gnpr_baseline.prepare_realworld_data import collapse_consecutive_same_poi


def spatial_features(latitude: float, longitude: float) -> np.ndarray:
    lat = math.radians(latitude)
    lon = math.radians(longitude)
    return np.asarray(
        [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        dtype=np.float32,
    )


def temporal_features(hour_counts: Counter[int]) -> np.ndarray:
    if not hour_counts:
        return np.zeros(12, dtype=np.float32)
    hours = np.asarray(sorted(hour_counts), dtype=np.float32)
    counts = np.asarray([hour_counts[int(hour)] for hour in hours], dtype=np.float32)
    weights = counts / counts.sum()
    values: list[float] = []
    for frequency in range(1, 7):
        angle = 2 * np.pi * frequency * hours / 24.0
        values.extend([float(np.sum(weights * np.sin(angle))), float(np.sum(weights * np.cos(angle)))])
    return np.asarray(values, dtype=np.float32)


def category_features(
    categories: list[str],
    model_path: str,
    output_dim: int,
    device: str,
) -> tuple[dict[str, np.ndarray], int]:
    model = SentenceTransformer(model_path, device=device)
    encoded = model.encode(categories, show_progress_bar=True, convert_to_numpy=True)
    effective_dim = min(output_dim, len(categories), int(encoded.shape[1]))
    reduced = PCA(n_components=effective_dim, random_state=2024).fit_transform(encoded)
    if effective_dim < output_dim:
        reduced = np.pad(reduced, ((0, 0), (0, output_dim - effective_dim)))
    return {
        category: reduced[index].astype(np.float32)
        for index, category in enumerate(categories)
    }, effective_dim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GNPR 79-dimensional POI embeddings.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--pois", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path, required=True)
    parser.add_argument("--train_end", required=True)
    parser.add_argument("--category_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timezone_offset_minutes", type=int, default=480)
    parser.add_argument("--category_dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mappings = json.loads(args.id_mappings.read_text(encoding="utf-8"))
    poi_map = {str(key): int(value) for key, value in mappings["poi_id_to_internal"].items()}

    events = pd.read_csv(args.events)
    pois = pd.read_csv(args.pois)
    events["_user"] = events["user_id"].astype(str)
    events["_poi"] = events["poi_id"].astype(str)
    events["_time"] = pd.to_datetime(events["timestamp"], utc=True, errors="raise")
    events = events[events["_poi"].isin(poi_map)].copy()
    events = events.drop_duplicates(subset=["_user", "_poi", "_time"])
    events, collapsed = collapse_consecutive_same_poi(events)

    train_end = pd.Timestamp(args.train_end)
    if train_end.tzinfo is None:
        raise ValueError("train_end 必须包含时区")
    train_end = train_end.tz_convert("UTC")
    events = events[events["_time"] <= train_end].copy()
    events["pid"] = events["_poi"].map(poi_map)
    events["local_hour"] = (
        events["_time"] + pd.Timedelta(minutes=args.timezone_offset_minutes)
    ).dt.hour

    pois["_poi"] = pois["poi_id"].astype(str)
    pois = pois[pois["_poi"].isin(poi_map)].copy()
    pois["pid"] = pois["_poi"].map(poi_map)
    pois = pois.sort_values("pid").reset_index(drop=True)
    if len(pois) != len(poi_map) or set(pois["pid"]) != set(poi_map.values()):
        raise ValueError("POI 元数据与统一预处理目录不一致")

    categories = sorted(pois["category_l2"].astype(str).unique())
    category_to_embedding, effective_dim = category_features(
        categories,
        args.category_model,
        args.category_dim,
        args.device,
    )
    hour_counts = {
        int(pid): Counter(int(hour) for hour in group["local_hour"])
        for pid, group in events.groupby("pid")
    }

    poi_vectors: dict[int, np.ndarray] = {}
    poi_rows: list[dict[str, object]] = []
    for row in pois.itertuples(index=False):
        pid = int(row.pid)
        category = str(row.category_l2)
        vector = np.concatenate(
            [
                category_to_embedding[category],
                spatial_features(float(row.latitude), float(row.longitude)),
                temporal_features(hour_counts.get(pid, Counter())),
            ]
        ).astype(np.float32)
        poi_vectors[pid] = vector
        poi_rows.append(
            {
                "pid": pid,
                "category": category,
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "visit_time_and_count": json.dumps(
                    dict(sorted(hour_counts.get(pid, Counter()).items()))
                ),
            }
        )

    with (args.output_dir / "poi_Emb_dict.pkl").open("wb") as handle:
        pickle.dump(poi_vectors, handle)
    with (args.output_dir / "category_to_embedding.pkl").open("wb") as handle:
        pickle.dump(category_to_embedding, handle)
    pd.DataFrame(poi_rows).to_csv(args.output_dir / "poi_info_for_sid.csv", index=False)
    report = {
        "status": "GNPR_POI_EMBEDDINGS_OK",
        "train_end_utc": train_end.isoformat(),
        "pois": len(poi_vectors),
        "categories": len(categories),
        "category_dim": args.category_dim,
        "category_pca_effective_dim": effective_dim,
        "spatial_dim": 3,
        "temporal_dim": 12,
        "total_dim": len(next(iter(poi_vectors.values()))),
        "train_state_events": len(events),
        "collapsed_consecutive_events": collapsed,
        "category_field": "category_l2",
        "category_model": args.category_model,
        "data_protocol": "reuse TAP preprocessing mappings and train cutoff",
    }
    (args.output_dir / "embedding_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
