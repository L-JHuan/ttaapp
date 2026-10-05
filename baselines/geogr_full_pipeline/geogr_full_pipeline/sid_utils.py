"""构建 GeoGR 风格的地理—协同对比表示与三层 RQ-Kmeans SID。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
SID_LABELS = "abcdefghijklmnopqrstuvwxyz"


@dataclass(frozen=True)
class GeoPair:
    left: int
    right: int
    score: float
    distance_km: float
    common_users: int


@dataclass(frozen=True)
class CatalogPoi:
    pid: int
    latitude: float
    longitude: float
    category_l1: str
    category_l2: str

    @property
    def description(self) -> str:
        return (
            f"Category level 1: {self.category_l1}; "
            f"category level 2: {self.category_l2}; "
            f"latitude: {self.latitude:.6f}; longitude: {self.longitude:.6f}."
        )


def haversine_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, left)
    lat2, lon2 = map(math.radians, right)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(value)))


def build_geo_constrained_pairs(
    user_items: dict[int, set[int]],
    coordinates: dict[int, tuple[float, float]],
    max_distance_km: float,
    min_common_users: int,
    swing_alpha: float,
    max_pairs_per_poi: int = 0,
) -> tuple[list[GeoPair], dict[str, int | float]]:
    """按 Swing 共访得分和地理距离构造训练集正样本对。"""
    if max_distance_km <= 0:
        raise ValueError("max_distance_km 必须为正数")
    if min_common_users < 2:
        raise ValueError("Swing 至少需要两个共同用户，min_common_users 必须 >= 2")
    if swing_alpha <= 0:
        raise ValueError("swing_alpha 必须为正数")

    pair_users: dict[tuple[int, int], set[int]] = defaultdict(set)
    for user_id, items in user_items.items():
        ordered = sorted(pid for pid in items if pid in coordinates)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                pair_users[(left, right)].add(user_id)

    candidates: list[GeoPair] = []
    behavior_candidate_pairs = 0
    for (left, right), common in pair_users.items():
        if len(common) < min_common_users:
            continue
        behavior_candidate_pairs += 1
        score = 0.0
        ordered_users = sorted(common)
        for first_index, first_user in enumerate(ordered_users):
            for second_user in ordered_users[first_index + 1 :]:
                overlap = len(user_items[first_user] & user_items[second_user])
                score += 1.0 / (swing_alpha + overlap)
        distance = haversine_km(coordinates[left], coordinates[right])
        if distance <= max_distance_km:
            candidates.append(
                GeoPair(
                    left=left,
                    right=right,
                    score=float(score),
                    distance_km=float(distance),
                    common_users=len(common),
                )
            )

    candidates.sort(
        key=lambda pair: (-pair.score, pair.distance_km, pair.left, pair.right)
    )
    if max_pairs_per_poi > 0:
        degrees: Counter[int] = Counter()
        selected: list[GeoPair] = []
        for pair in candidates:
            if (
                degrees[pair.left] >= max_pairs_per_poi
                or degrees[pair.right] >= max_pairs_per_poi
            ):
                continue
            selected.append(pair)
            degrees[pair.left] += 1
            degrees[pair.right] += 1
        candidates = selected

    report: dict[str, int | float] = {
        "users": len(user_items),
        "behavior_candidate_pairs": behavior_candidate_pairs,
        "geo_retained_pairs": len(candidates),
        "max_distance_km": float(max_distance_km),
        "min_common_users": int(min_common_users),
        "swing_alpha": float(swing_alpha),
        "max_pairs_per_poi": int(max_pairs_per_poi),
    }
    return candidates, report


def residual_kmeans(
    embeddings: np.ndarray,
    codebook_size: int,
    num_layers: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """按照论文公式逐层对残差执行 K-means 量化。"""
    from sklearn.cluster import KMeans

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError(f"embeddings 形状非法: {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("embeddings 包含 NaN 或 Inf")
    if codebook_size <= 1 or num_layers <= 0:
        raise ValueError("codebook_size 必须大于 1，num_layers 必须为正数")

    residual = values.copy()
    quantized = np.zeros_like(values)
    layer_codes: list[np.ndarray] = []
    layer_reports: list[dict[str, float | int]] = []
    for layer in range(num_layers):
        clusters = min(codebook_size, len(values))
        model = KMeans(
            n_clusters=clusters,
            random_state=seed + layer,
            n_init=20,
        )
        codes = model.fit_predict(residual).astype(np.int64)
        selected = model.cluster_centers_[codes].astype(np.float32)
        residual = residual - selected
        quantized = quantized + selected
        layer_codes.append(codes)
        layer_reports.append(
            {
                "layer": layer + 1,
                "configured_codebook_size": int(codebook_size),
                "used_codes": int(len(set(codes.tolist()))),
                "mean_residual_l2": float(np.linalg.norm(residual, axis=1).mean()),
            }
        )
    return (
        np.stack(layer_codes, axis=1),
        quantized,
        {"layers": layer_reports},
    )


def add_collision_leaf(codes: dict[int, list[int]]) -> dict[int, list[int]]:
    counts = Counter(tuple(code) for code in codes.values())
    next_leaf: Counter[tuple[int, ...]] = Counter()
    result: dict[int, list[int]] = {}
    for pid in sorted(codes):
        code = list(codes[pid])
        key = tuple(code)
        if counts[key] > 1:
            code.append(next_leaf[key])
            next_leaf[key] += 1
        result[pid] = code
    return result


def _inverse_mapping(mapping: dict[str, int]) -> dict[int, str]:
    result = {int(value): str(key) for key, value in mapping.items()}
    if len(result) != len(mapping):
        raise ValueError("类别映射不是一一对应")
    return result


def _find_frame_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    by_lower = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(f"缺少字段 {candidates}; 当前字段={list(frame.columns)}")


def load_catalog(
    poi_info_path: Path,
    role_priors_path: Path,
    id_mappings_path: Path | None = None,
) -> list[CatalogPoi]:
    poi = pd.read_csv(poi_info_path)
    role = pd.read_csv(role_priors_path)
    poi = poi.rename(
        columns={
            _find_frame_column(poi, ("pid", "PoiId", "poi_id")): "pid",
            _find_frame_column(poi, ("latitude", "Latitude", "lat")): "latitude",
            _find_frame_column(
                poi,
                ("longitude", "Longitude", "lon", "lng"),
            ): "longitude",
        }
    )
    role = role.rename(
        columns={_find_frame_column(role, ("pid", "PoiId", "poi_id")): "pid"}
    )
    required_role = {"pid", "l1_label", "l2_label"}
    if missing := sorted(required_role - set(role.columns)):
        raise ValueError(f"{role_priors_path} 缺少字段: {missing}")
    if poi["pid"].duplicated().any() or role["pid"].duplicated().any():
        raise ValueError("POI 目录或类别表包含重复 pid")

    category_columns_available = {"category_l1", "category_l2"}.issubset(role.columns)
    if category_columns_available:
        l1_names: dict[int, str] = {}
        l2_names: dict[int, str] = {}
    elif id_mappings_path is not None:
        mappings = json.loads(id_mappings_path.read_text(encoding="utf-8"))
        l1_names = _inverse_mapping(mappings["category_l1_to_internal"])
        l2_names = _inverse_mapping(mappings["category_l2_to_internal"])
    else:
        l1_names = {}
        l2_names = {}
    role_columns = ["pid", "l1_label", "l2_label"]
    if category_columns_available:
        role_columns.extend(["category_l1", "category_l2"])
    merged = poi.merge(
        role[role_columns],
        on="pid",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(poi):
        raise ValueError("POI 目录与类别表覆盖范围不一致")
    catalog: list[CatalogPoi] = []
    for row in merged.sort_values("pid").itertuples(index=False):
        l1 = int(row.l1_label)
        l2 = int(row.l2_label)
        if category_columns_available:
            category_l1 = str(row.category_l1)
            category_l2 = str(row.category_l2)
        elif l1_names and l2_names:
            if l1 not in l1_names or l2 not in l2_names:
                raise ValueError(f"pid={row.pid} 的类别映射不存在")
            category_l1 = l1_names[l1]
            category_l2 = l2_names[l2]
        else:
            category_l1 = f"L1-{l1}"
            category_l2 = f"L2-{l2}"
        catalog.append(
            CatalogPoi(
                pid=int(row.pid),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                category_l1=category_l1,
                category_l2=category_l2,
            )
        )
    return catalog


def load_train_user_items(path: Path) -> dict[int, set[int]]:
    frame = pd.read_csv(path)
    required = {"UserId", "sequence_PoiId"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{path} 缺少字段: {missing}")
    result: dict[int, set[int]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        sequence = ast.literal_eval(str(row.sequence_PoiId))
        if not isinstance(sequence, (list, tuple)):
            raise ValueError(f"UserId={row.UserId} 的 sequence_PoiId 非列表")
        result[int(row.UserId)].update(int(pid) for pid in sequence)
    return dict(result)


def encode_descriptions(
    catalog: list[CatalogPoi],
    encoder_model: str,
    device: str,
    backend: str,
    dtype_name: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """使用给定 Hugging Face 编码器生成初始 POI 描述向量。"""
    import torch

    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name == "auto":
        model_dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    elif dtype_name in dtype_by_name:
        model_dtype = dtype_by_name[dtype_name]
    else:
        raise ValueError(
            "encoder_dtype 必须是 auto、float32、float16 或 bfloat16"
        )

    descriptions = [poi.description for poi in catalog]
    if backend == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            encoder_model,
            device=device,
            model_kwargs={"torch_dtype": model_dtype},
        )
        model.max_seq_length = max_length
        embeddings = model.encode(
            descriptions,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
    if backend not in {"transformers_mean", "transformers_last_token"}:
        raise ValueError(
            "encoder_backend 必须是 sentence_transformers、transformers_mean "
            "或 transformers_last_token"
        )

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(encoder_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    model = AutoModel.from_pretrained(
        encoder_model,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(descriptions), batch_size):
            batch = tokenizer(
                descriptions[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch, return_dict=True).last_hidden_state
            attention_mask = batch["attention_mask"]
            if backend == "transformers_last_token":
                last_indices = (
                    attention_mask.size(1)
                    - 1
                    - attention_mask.flip(dims=(1,)).argmax(dim=1)
                )
                pooled = hidden[
                    torch.arange(hidden.size(0), device=hidden.device),
                    last_indices,
                ]
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def load_initial_embeddings(path: Path, expected_pids: list[int]) -> np.ndarray:
    payload = np.load(path)
    if "pids" not in payload or "embeddings" not in payload:
        raise ValueError(f"{path} 必须包含 pids 和 embeddings")
    pids = [int(value) for value in payload["pids"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if len(pids) != len(embeddings) or len(set(pids)) != len(pids):
        raise ValueError("初始向量的 pid 与矩阵行数不一致或 pid 重复")
    index = {pid: row for row, pid in enumerate(pids)}
    missing = [pid for pid in expected_pids if pid not in index]
    if missing:
        raise ValueError(f"初始向量缺少 {len(missing)} 个目录 POI: {missing[:10]}")
    return np.stack([embeddings[index[pid]] for pid in expected_pids], axis=0)


def contrastive_project(
    embeddings: np.ndarray,
    pairs: list[GeoPair],
    pid_to_index: dict[int, int],
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """用线性投影适配层近似论文中的 P2P 对比微调。"""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    if not pairs:
        raise ValueError("没有可用于对比学习的地理约束共访 POI 对")
    if epochs <= 0 or batch_size <= 1 or temperature <= 0:
        raise ValueError("对比学习参数非法")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    values = torch.tensor(embeddings, dtype=torch.float32, device=device)
    dimension = values.shape[1]
    projection = nn.Linear(dimension, dimension, bias=False, device=device)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(dimension, device=device))
    optimizer = torch.optim.AdamW(projection.parameters(), lr=learning_rate)
    pair_indices = [(pid_to_index[pair.left], pid_to_index[pair.right]) for pair in pairs]
    generator = random.Random(seed)
    epoch_losses: list[float] = []
    projection.train()
    for _ in range(epochs):
        generator.shuffle(pair_indices)
        losses: list[float] = []
        for start in range(0, len(pair_indices), batch_size):
            batch = pair_indices[start : start + batch_size]
            if len(batch) < 2:
                continue
            left = torch.tensor([item[0] for item in batch], device=device)
            right = torch.tensor([item[1] for item in batch], device=device)
            left_vector = functional.normalize(projection(values[left]), dim=-1)
            right_vector = functional.normalize(projection(values[right]), dim=-1)
            logits = left_vector @ right_vector.T / temperature
            labels = torch.arange(len(batch), device=device)
            loss = 0.5 * (
                functional.cross_entropy(logits, labels)
                + functional.cross_entropy(logits.T, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("每个对比学习 batch 至少需要两个正样本对")
        epoch_losses.append(float(np.mean(losses)))

    projection.eval()
    with torch.no_grad():
        refined = functional.normalize(projection(values), dim=-1).cpu().numpy()
    return refined.astype(np.float32), {
        "adapter": "trainable linear projection over fixed initial encoder embeddings",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "temperature": float(temperature),
        "epoch_losses": epoch_losses,
    }


def sid_tokens(code: Iterable[int]) -> str:
    values = list(code)
    if len(values) > len(SID_LABELS):
        raise ValueError("SID 层数过多")
    return "".join(f"<{SID_LABELS[index]}_{value}>" for index, value in enumerate(values))


def write_codebook(
    path: Path,
    pids: list[int],
    codes: dict[int, list[int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "sid", "sid_tokens"])
        writer.writeheader()
        for pid in pids:
            writer.writerow(
                {
                    "pid": pid,
                    "sid": str(codes[pid]),
                    "sid_tokens": sid_tokens(codes[pid]),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paper-guided GeoGR SID under the matched TAP-SID protocol."
    )
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--train_sequences", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--initial_embeddings_npz", type=Path)
    parser.add_argument("--save_initial_embeddings_npz", type=Path)
    parser.add_argument("--encoder_model", default="")
    parser.add_argument("--encoder_device", default="cuda:0")
    parser.add_argument("--encoder_backend", default="sentence_transformers")
    parser.add_argument("--encoder_dtype", default="auto")
    parser.add_argument("--encoder_batch_size", type=int, default=16)
    parser.add_argument("--encoder_max_length", type=int, default=128)
    parser.add_argument("--max_distance_km", type=float, default=3.0)
    parser.add_argument("--min_common_users", type=int, default=2)
    parser.add_argument("--swing_alpha", type=float, default=1.0)
    parser.add_argument("--max_pairs_per_poi", type=int, default=50)
    parser.add_argument("--contrastive_device", default="cuda:0")
    parser.add_argument("--contrastive_epochs", type=int, default=10)
    parser.add_argument("--contrastive_batch_size", type=int, default=256)
    parser.add_argument("--contrastive_learning_rate", type=float, default=1e-4)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.poi_info, args.role_priors, args.id_mappings)
    pids = [poi.pid for poi in catalog]
    pid_to_index = {pid: index for index, pid in enumerate(pids)}
    coordinates = {poi.pid: (poi.latitude, poi.longitude) for poi in catalog}
    user_items = load_train_user_items(args.train_sequences)
    pairs, pair_report = build_geo_constrained_pairs(
        user_items,
        coordinates,
        max_distance_km=args.max_distance_km,
        min_common_users=args.min_common_users,
        swing_alpha=args.swing_alpha,
        max_pairs_per_poi=args.max_pairs_per_poi,
    )

    if args.initial_embeddings_npz is not None:
        initial = load_initial_embeddings(args.initial_embeddings_npz, pids)
        embedding_source = str(args.initial_embeddings_npz)
    else:
        if not args.encoder_model:
            raise ValueError("未提供 initial_embeddings_npz 时必须设置 encoder_model")
        initial = encode_descriptions(
            catalog,
            encoder_model=args.encoder_model,
            device=args.encoder_device,
            backend=args.encoder_backend,
            dtype_name=args.encoder_dtype,
            batch_size=args.encoder_batch_size,
            max_length=args.encoder_max_length,
        )
        embedding_source = args.encoder_model
    if args.save_initial_embeddings_npz is not None:
        args.save_initial_embeddings_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_initial_embeddings_npz,
            pids=np.asarray(pids, dtype=np.int64),
            embeddings=initial,
        )

    refined, contrastive_report = contrastive_project(
        initial,
        pairs,
        pid_to_index,
        device=args.contrastive_device,
        epochs=args.contrastive_epochs,
        batch_size=args.contrastive_batch_size,
        learning_rate=args.contrastive_learning_rate,
        temperature=args.contrastive_temperature,
        seed=args.seed,
    )
    raw_codes_array, _quantized, quantization_report = residual_kmeans(
        refined,
        codebook_size=args.codebook_size,
        num_layers=args.num_layers,
        seed=args.seed,
    )
    raw_codes = {
        pid: [int(value) for value in raw_codes_array[row_index].tolist()]
        for row_index, pid in enumerate(pids)
    }
    final_codes = add_collision_leaf(raw_codes)
    raw_counts = Counter(tuple(code) for code in raw_codes.values())
    write_codebook(args.output_csv, pids, final_codes)

    report = {
        "status": "GEOGR_ADAPTED_CODEBOOK_OK",
        "implementation": "paper-guided current-protocol adaptation",
        "paper_components": {
            "geo_constrained_swing_pairs": "implemented",
            "poi_to_poi_nce": "implemented as a trainable projection over fixed encoder embeddings",
            "rq_kmeans_layers": int(args.num_layers),
            "em_style_sid_optimization": (
                "not reproduced because the paper does not specify the collision-free "
                "assignment procedure, iteration count, or optimization corpus"
            ),
        },
        "catalog_pois": len(pids),
        "embedding_source": embedding_source,
        "encoder_backend": args.encoder_backend,
        "encoder_dtype": args.encoder_dtype,
        "embedding_dimension": int(initial.shape[1]),
        "pair_construction": pair_report,
        "contrastive_training": contrastive_report,
        "quantization": quantization_report,
        "raw_unique_paths": len(raw_counts),
        "raw_colliding_pois": int(sum(count for count in raw_counts.values() if count > 1)),
        "final_unique_paths": len({tuple(code) for code in final_codes.values()}),
        "collision_leaf_rule": "append 0..n-1 by ascending pid within duplicate RQ paths",
        "codebook_size": int(args.codebook_size),
        "seed": int(args.seed),
        "output_csv": str(args.output_csv),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
