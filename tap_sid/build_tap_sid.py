from __future__ import annotations

import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"missing column candidates={candidates}, columns={list(df.columns)}")


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    return (values - values.mean()) / (values.std() + 1e-12)


def fourier_coord_features(lat_norm: np.ndarray, lon_norm: np.ndarray) -> np.ndarray:
    feats = [lat_norm, lon_norm]
    for base in [lat_norm, lon_norm]:
        for freq in [1, 2, 4, 8]:
            angle = 2.0 * math.pi * freq * base
            feats.append(np.sin(angle))
            feats.append(np.cos(angle))
    return np.stack(feats, axis=1).astype(np.float32)


def allocate_codes_by_sqrt(counts: dict[int, int], total_codes: int) -> dict[int, int]:
    labels = sorted(counts)
    weights = np.array([math.sqrt(max(counts[label], 1)) for label in labels], dtype=np.float64)
    raw = weights / weights.sum() * total_codes
    alloc = np.maximum(1, np.floor(raw).astype(int))
    while alloc.sum() < total_codes:
        remainders = raw - np.floor(raw)
        for idx in np.argsort(-remainders):
            if alloc.sum() >= total_codes:
                break
            alloc[idx] += 1
    while alloc.sum() > total_codes:
        removable = [idx for idx, value in enumerate(alloc) if value > 1]
        if not removable:
            break
        remainders = raw - np.floor(raw)
        idx = min(removable, key=lambda i: remainders[i])
        alloc[idx] -= 1
    return {label: int(value) for label, value in zip(labels, alloc)}


def build_coarse_fine_region_codes(
    poi: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    n_coarse_regions: int,
    n_fine_regions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_norm = normalize(poi[lat_col].to_numpy())
    lon_norm = normalize(poi[lon_col].to_numpy())
    xy = np.stack([lat_norm, lon_norm], axis=1)
    n_coarse_regions = min(n_coarse_regions, len(poi))
    n_fine_regions = min(n_fine_regions, len(poi))

    coarse_model = KMeans(n_clusters=n_coarse_regions, random_state=seed, n_init=20)
    coarse_ids = coarse_model.fit_predict(xy).astype(np.int64)
    coord_feat = fourier_coord_features(lat_norm, lon_norm)

    coarse_counts = Counter(int(x) for x in coarse_ids)
    alloc = allocate_codes_by_sqrt(coarse_counts, n_fine_regions)

    fine_ids = np.zeros(len(poi), dtype=np.int64)
    next_fine_code = 0
    for label in sorted(alloc):
        idx = np.where(coarse_ids == label)[0]
        k = min(alloc[label], len(idx))
        if k <= 1:
            fine_ids[idx] = next_fine_code
            next_fine_code += 1
            continue
        local = KMeans(n_clusters=k, random_state=seed + int(label) + 1, n_init=20)
        local_assign = local.fit_predict(coord_feat[idx])
        for local_id in range(k):
            fine_ids[idx[local_assign == local_id]] = next_fine_code + local_id
        next_fine_code += k
    return coarse_ids, fine_ids, coord_feat


def bucket_stats(values: list[tuple[int, ...]]) -> dict:
    counts = Counter(values)
    sizes = np.array(list(counts.values()), dtype=np.float64)
    return {
        "bucket_count": int(len(counts)),
        "mean_bucket_size": float(sizes.mean()) if len(sizes) else 0.0,
        "median_bucket_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "p90_bucket_size": float(np.percentile(sizes, 90)) if len(sizes) else 0.0,
        "max_bucket_size": int(sizes.max()) if len(sizes) else 0,
        "singletons": int((sizes == 1).sum()) if len(sizes) else 0,
        "collision_buckets": int((sizes > 1).sum()) if len(sizes) else 0,
        "collision_poi_count": int(sizes[sizes > 1].sum()) if len(sizes) else 0,
    }


def safe_literal_vector(value: str) -> list[float]:
    try:
        parsed = ast.literal_eval(str(value))
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build TAP-SID SID: coarse region a + fine region b + label c/d + leaf e."
    )
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--n_coarse_regions", type=int, default=64)
    parser.add_argument("--n_fine_regions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    poi = pd.read_csv(args.poi_info)
    pid_col = find_column(poi, ["pid", "PoiId", "poi_id"])
    lat_col = find_column(poi, ["latitude", "Latitude", "lat"])
    lon_col = find_column(poi, ["longitude", "Longitude", "lon", "lng"])
    poi = poi.rename(columns={pid_col: "pid"}).copy()
    poi["pid"] = poi["pid"].astype(int)
    poi = poi.sort_values("pid").reset_index(drop=True)

    role = pd.read_csv(args.role_priors)
    role_pid_col = find_column(role, ["pid", "PoiId", "poi_id"])
    role = role.rename(columns={role_pid_col: "pid"}).copy()
    for col in ["l1_label", "l2_label"]:
        if col not in role.columns:
            raise ValueError(f"{args.role_priors} missing required column {col}")
    role = role[["pid", "l1_label", "l2_label"]].copy()
    role["pid"] = role["pid"].astype(int)
    role["l1_label"] = role["l1_label"].astype(int)
    role["l2_label"] = role["l2_label"].astype(int)

    df = poi.merge(role, on="pid", how="inner")
    if len(df) != len(poi):
        missing = sorted(set(poi["pid"]) - set(df["pid"]))
        raise ValueError(f"role_priors missing {len(missing)} POIs, examples={missing[:10]}")

    coarse_ids, fine_ids, coord_feat = build_coarse_fine_region_codes(
        df,
        lat_col=lat_col,
        lon_col=lon_col,
        n_coarse_regions=args.n_coarse_regions,
        n_fine_regions=args.n_fine_regions,
        seed=args.seed,
    )
    df["a_id"] = coarse_ids
    df["b_id"] = fine_ids
    df["c_id"] = df["l1_label"].astype(int)
    df["d_id"] = df["l2_label"].astype(int)

    d_ids = []
    for _, group in df.groupby(["a_id", "b_id", "c_id", "d_id"], sort=False):
        ordered = group.sort_values("pid")
        for local_id, row_idx in enumerate(ordered.index):
            d_ids.append((row_idx, local_id))
    d_series = pd.Series({idx: local_id for idx, local_id in d_ids})
    df["e_id"] = d_series.loc[df.index].astype(int).to_numpy()

    sid_values = []
    sid_tokens = []
    for row in df.itertuples(index=False):
        sid = [int(row.a_id), int(row.b_id), int(row.c_id), int(row.d_id), int(row.e_id)]
        sid_values.append(sid)
        sid_tokens.append(f"<a_{sid[0]}><b_{sid[1]}><c_{sid[2]}><d_{sid[3]}><e_{sid[4]}>")
    df["sid"] = sid_values
    df["sid_tokens"] = sid_tokens

    out = pd.DataFrame(
        {
            "pid": df["pid"].astype(int),
            "sid": df["sid"].astype(str),
            "sid_tokens": df["sid_tokens"],
            "a_id": df["a_id"].astype(int),
            "b_id": df["b_id"].astype(int),
            "c_id": df["c_id"].astype(int),
            "d_id": df["d_id"].astype(int),
            "e_id": df["e_id"].astype(int),
            "coarse_region_id": df["a_id"].astype(int),
            "fine_region_id": df["b_id"].astype(int),
            "l1_label": df["l1_label"].astype(int),
            "l2_label": df["l2_label"].astype(int),
        }
    ).sort_values("pid")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    prefix_a = [(int(x),) for x in df["a_id"]]
    prefix_ab = [(int(a), int(b)) for a, b in zip(df["a_id"], df["b_id"])]
    prefix_abc = [
        (int(a), int(b), int(c)) for a, b, c in zip(df["a_id"], df["b_id"], df["c_id"])
    ]
    prefix_abcd = [
        (int(a), int(b), int(c), int(d))
        for a, b, c, d in zip(df["a_id"], df["b_id"], df["c_id"], df["d_id"])
    ]
    full = [
        (int(a), int(b), int(c), int(d), int(e))
        for a, b, c, d, e in zip(df["a_id"], df["b_id"], df["c_id"], df["d_id"], df["e_id"])
    ]
    prefix_counts = Counter(prefix_abcd)
    e_nonzero = int((df["e_id"] != 0).sum())
    report = {
        "name": "tap_sid_coarse_fine_region_sid",
        "poi_info": str(args.poi_info),
        "role_priors": str(args.role_priors),
        "output_csv": str(args.output_csv),
        "num_pois": int(len(df)),
        "n_coarse_regions": int(args.n_coarse_regions),
        "n_fine_regions": int(args.n_fine_regions),
        "used_a_codes": int(df["a_id"].nunique()),
        "used_b_codes": int(df["b_id"].nunique()),
        "used_c_codes": int(df["c_id"].nunique()),
        "used_d_codes": int(df["d_id"].nunique()),
        "used_e_codes": int(df["e_id"].nunique()),
        "max_e_id": int(df["e_id"].max()),
        "e_nonzero_poi_count": e_nonzero,
        "e_nonzero_ratio": float(e_nonzero / len(df)),
        "prefix_bucket_stats": {
            "a": bucket_stats(prefix_a),
            "ab": bucket_stats(prefix_ab),
            "abc": bucket_stats(prefix_abc),
            "abcd": bucket_stats(prefix_abcd),
            "abcde": bucket_stats(full),
        },
        "top_abcd_buckets": [
            {"prefix": list(prefix), "size": int(size)}
            for prefix, size in prefix_counts.most_common(20)
        ],
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved csv: {args.output_csv}")
    print(f"saved report: {args.report_json}")


if __name__ == "__main__":
    main()
