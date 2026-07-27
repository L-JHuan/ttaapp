#!/usr/bin/env python3
"""按目标 POI 所在城市统计 TAP-SID 与 Residual SID 的推荐指标。"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


METRIC_NAMES = ("R@1", "R@5", "R@10", "NDCG@10")
METRIC_JSON_KEYS = ("recall@1", "recall@5", "recall@10", "ndcg@10")
DIRECT_CITIES = {
    110000: "北京市",
    120000: "天津市",
    310000: "上海市",
    500000: "重庆市",
    810000: "香港特别行政区",
    820000: "澳门特别行政区",
}
QUERY_RE = re.compile(
    r"When\s+(?P<time>.+?)\s+user_(?P<user>\d+)\s+is likely to visit:",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tap-run-root", type=Path, required=True)
    parser.add_argument("--residual-run-root", type=Path, required=True)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--province-geojson", type=Path, required=True)
    parser.add_argument("--city-boundary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} 不存在或为空：{path}")
    return path


def load_json(path: Path) -> Any:
    with require_file(path, "JSON 文件").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sid_list_to_tokens(values: Iterable[int]) -> str:
    return "".join(
        f"<{chr(ord('a') + index)}_{int(value)}>"
        for index, value in enumerate(values)
    )


def load_sid_to_pid(path: Path, method: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with require_file(path, f"{method} 码本").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            pid = str(row["pid"])
            sid = row.get("sid_tokens", "").strip()
            if not sid:
                raw_sid = row.get("sid", "").strip()
                if raw_sid:
                    sid = sid_list_to_tokens(ast.literal_eval(raw_sid))
            if not sid:
                raise ValueError(f"{method} 码本缺少 sid_tokens/sid：pid={pid}")
            if sid in mapping:
                raise ValueError(f"{method} 码本存在重复完整 SID：{sid}")
            mapping[sid] = pid
    if not mapping:
        raise ValueError(f"{method} 码本为空：{path}")
    return mapping


def load_poi_coordinates(path: Path) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}
    with require_file(path, "POI 信息").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            coordinates[str(row["pid"])] = (
                float(row["longitude"]),
                float(row["latitude"]),
            )
    return coordinates


def load_test_targets(path: Path) -> list[str]:
    targets: list[str] = []
    with require_file(path, "测试序列").open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            sequence = ast.literal_eval(row["sequence_PoiId"])
            if not sequence:
                raise ValueError("测试序列中存在空 POI 轨迹")
            targets.append(str(sequence[-1]))
    return targets


def parse_query_key(text: str) -> tuple[str, str]:
    match = QUERY_RE.search(text)
    if not match:
        raise ValueError(f"无法解析查询用户与时间：{text[-180:]}")
    return match.group("user"), match.group("time")


def polygon_bbox(coordinates: Any) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
            xs.append(float(obj[0]))
            ys.append(float(obj[1]))
            return
        if isinstance(obj, list):
            for child in obj:
                collect(child)

    collect(coordinates)
    if not xs:
        raise ValueError("行政区边界包含空几何对象")
    return min(xs), min(ys), max(xs), max(ys)


def point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    eps: float = 1e-10,
) -> bool:
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross) > eps:
        return False
    return (
        min(x1, x2) - eps <= x <= max(x1, x2) + eps
        and min(y1, y2) - eps <= y <= max(y1, y2) + eps
    )


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    if len(ring) < 3:
        return False
    inside = False
    previous = len(ring) - 1
    for current in range(len(ring)):
        x1, y1 = float(ring[previous][0]), float(ring[previous][1])
        x2, y2 = float(ring[current][0]), float(ring[current][1])
        if point_on_segment(x, y, x1, y1, x2, y2):
            return True
        if (y1 > y) != (y2 > y):
            cross_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross_x:
                inside = not inside
        previous = current
    return inside


def point_in_polygon(
    x: float,
    y: float,
    polygon: list[list[list[float]]],
) -> bool:
    if not polygon or not point_in_ring(x, y, polygon[0]):
        return False
    return not any(point_in_ring(x, y, hole) for hole in polygon[1:])


def point_in_geometry(x: float, y: float, geometry: dict[str, Any]) -> bool:
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    if geometry_type == "Polygon":
        return point_in_polygon(x, y, coordinates)
    if geometry_type == "MultiPolygon":
        return any(
            point_in_polygon(x, y, polygon)
            for polygon in coordinates
        )
    raise ValueError(f"不支持的行政区几何类型：{geometry_type}")


def load_city_boundaries(
    province_geojson: Path,
    city_boundary_dir: Path,
) -> list[dict[str, Any]]:
    if not city_boundary_dir.is_dir():
        raise FileNotFoundError(f"城市边界目录不存在：{city_boundary_dir}")
    boundaries: list[dict[str, Any]] = []

    provinces = load_json(province_geojson)
    for feature in provinces["features"]:
        properties = feature["properties"]
        raw_adcode = str(properties["adcode"])
        if not raw_adcode.isdigit():
            continue
        adcode = int(raw_adcode)
        if adcode not in DIRECT_CITIES:
            continue
        geometry = feature["geometry"]
        boundaries.append(
            {
                "adcode": adcode,
                "city": DIRECT_CITIES[adcode],
                "geometry": geometry,
                "bbox": polygon_bbox(geometry["coordinates"]),
            }
        )

    for path in sorted(city_boundary_dir.glob("*_full.json")):
        collection = load_json(path)
        for feature in collection["features"]:
            properties = feature["properties"]
            geometry = feature["geometry"]
            boundaries.append(
                {
                    "adcode": int(properties["adcode"]),
                    "city": str(properties["name"]),
                    "geometry": geometry,
                    "bbox": polygon_bbox(geometry["coordinates"]),
                }
            )
    if not boundaries:
        raise ValueError("没有读取到任何城市行政区边界")
    return boundaries


def assign_city(
    longitude: float,
    latitude: float,
    boundaries: list[dict[str, Any]],
) -> tuple[str, int | None]:
    matches: list[tuple[str, int]] = []
    for boundary in boundaries:
        min_x, min_y, max_x, max_y = boundary["bbox"]
        if not (min_x <= longitude <= max_x and min_y <= latitude <= max_y):
            continue
        if point_in_geometry(longitude, latitude, boundary["geometry"]):
            matches.append((boundary["city"], boundary["adcode"]))
    if not matches:
        return "未匹配", None
    matches.sort(key=lambda item: item[1])
    return matches[0]


def rank_metrics(prediction: dict[str, Any], method: str, index: int) -> np.ndarray:
    gold = prediction["gold"]
    ranked = prediction["predictions"]
    if not isinstance(ranked, list) or len(ranked) < 5:
        raise ValueError(
            f"{method} 第 {index} 条预测不足 5 个候选："
            f"{len(ranked) if isinstance(ranked, list) else '非列表'}"
        )
    try:
        rank = ranked.index(gold) + 1
    except ValueError:
        rank = math.inf
    return np.asarray(
        [
            float(rank <= 1),
            float(rank <= 5),
            float(rank <= 10),
            1.0 / math.log2(rank + 1) if rank <= 10 else 0.0,
        ],
        dtype=np.float64,
    )


def official_metrics(path: Path) -> np.ndarray:
    metrics = load_json(path)
    return np.asarray(
        [float(metrics[key]) for key in METRIC_JSON_KEYS],
        dtype=np.float64,
    )


def bootstrap_city(
    tap_metrics: np.ndarray,
    residual_metrics: np.ndarray,
    users: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> dict[str, list[float]]:
    user_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, user in enumerate(users):
        user_to_indices[user].append(index)
    unique_users = sorted(user_to_indices)
    differences = np.zeros((iterations, len(METRIC_NAMES)), dtype=np.float64)
    relative = np.zeros((iterations, len(METRIC_NAMES)), dtype=np.float64)

    for iteration in range(iterations):
        sampled_users = rng.choice(
            unique_users,
            size=len(unique_users),
            replace=True,
        )
        sampled_indices: list[int] = []
        for user in sampled_users:
            sampled_indices.extend(user_to_indices[str(user)])
        tap_mean = tap_metrics[sampled_indices].mean(axis=0)
        residual_mean = residual_metrics[sampled_indices].mean(axis=0)
        differences[iteration] = tap_mean - residual_mean
        relative[iteration] = np.divide(
            tap_mean - residual_mean,
            residual_mean,
            out=np.full_like(tap_mean, np.nan),
            where=residual_mean != 0,
        )

    return {
        "diff_low": np.nanpercentile(differences, 2.5, axis=0).tolist(),
        "diff_high": np.nanpercentile(differences, 97.5, axis=0).tolist(),
        "rel_low": (
            100 * np.nanpercentile(relative, 2.5, axis=0)
        ).tolist(),
        "rel_high": (
            100 * np.nanpercentile(relative, 97.5, axis=0)
        ).tolist(),
    }


def metric_record(
    city: str,
    indices: list[int],
    tap_metrics: np.ndarray,
    residual_metrics: np.ndarray,
    users: list[str],
    target_pids: list[str],
    bootstrap_iterations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    tap = tap_metrics[selected]
    residual = residual_metrics[selected]
    tap_mean = tap.mean(axis=0)
    residual_mean = residual.mean(axis=0)
    absolute = tap_mean - residual_mean
    relative = np.divide(
        absolute,
        residual_mean,
        out=np.full_like(absolute, np.nan),
        where=residual_mean != 0,
    )
    selected_users = [users[index] for index in indices]
    selected_pids = [target_pids[index] for index in indices]
    bootstrap = bootstrap_city(
        tap,
        residual,
        selected_users,
        bootstrap_iterations,
        rng,
    )

    record: dict[str, Any] = {
        "city": city,
        "samples": len(indices),
        "users": len(set(selected_users)),
        "target_pois": len(set(selected_pids)),
    }
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        key = metric_name.replace("@", "").replace("NDCG", "N")
        record[f"residual_{key}"] = float(residual_mean[metric_index])
        record[f"tap_{key}"] = float(tap_mean[metric_index])
        record[f"abs_{key}"] = float(absolute[metric_index])
        record[f"rel_{key}_pct"] = finite_or_none(
            100 * relative[metric_index]
        )
        record[f"abs_{key}_ci_low"] = float(
            bootstrap["diff_low"][metric_index]
        )
        record[f"abs_{key}_ci_high"] = float(
            bootstrap["diff_high"][metric_index]
        )
        record[f"rel_{key}_ci_low_pct"] = finite_or_none(
            bootstrap["rel_low"][metric_index]
        )
        record[f"rel_{key}_ci_high_pct"] = finite_or_none(
            bootstrap["rel_high"][metric_index]
        )
    return record


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def resolve_inputs(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "tap_sid": args.tap_run_root / "codebook" / "tap_sid.csv",
        "residual_sid": (
            args.residual_run_root / "codebook" / "gnpr_sid.csv"
        ),
        "tap_test": args.tap_run_root / "data" / "llm_test.json",
        "residual_test": (
            args.residual_run_root / "data" / "llm_test.json"
        ),
        "tap_predictions": (
            args.tap_run_root / "eval" / "test_predictions.json"
        ),
        "residual_predictions": (
            args.residual_run_root / "eval" / "test_predictions.json"
        ),
        "tap_metrics": args.tap_run_root / "eval" / "test_metrics.json",
        "residual_metrics": (
            args.residual_run_root / "eval" / "test_metrics.json"
        ),
        "poi_info": args.processed_root / "poi_info.csv",
        "test_sequences": (
            args.processed_root
            / "v1_sequence"
            / "test_poi_sequence.csv"
        ),
        "province_geojson": args.province_geojson,
    }
    for label, path in paths.items():
        require_file(path, label)
    return paths


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap 必须为正整数")
    paths = resolve_inputs(args)

    tap_sid_to_pid = load_sid_to_pid(paths["tap_sid"], "TAP-SID")
    residual_sid_to_pid = load_sid_to_pid(
        paths["residual_sid"],
        "Residual SID",
    )
    coordinates = load_poi_coordinates(paths["poi_info"])
    test_targets = load_test_targets(paths["test_sequences"])
    tap_test = load_json(paths["tap_test"])
    residual_test = load_json(paths["residual_test"])
    tap_predictions = load_json(paths["tap_predictions"])
    residual_predictions = load_json(paths["residual_predictions"])

    lengths = {
        len(test_targets),
        len(tap_test),
        len(residual_test),
        len(tap_predictions),
        len(residual_predictions),
    }
    if len(lengths) != 1:
        raise ValueError(f"测试与预测样本数不一致：{sorted(lengths)}")
    sample_count = len(test_targets)
    if sample_count == 0:
        raise ValueError("测试集为空")

    users: list[str] = []
    target_pids: list[str] = []
    tap_metric_rows: list[np.ndarray] = []
    residual_metric_rows: list[np.ndarray] = []
    query_mismatches = 0
    target_mismatches = 0
    prediction_gold_mismatches = 0

    for index in range(sample_count):
        tap_key = parse_query_key(tap_test[index]["input"])
        residual_key = parse_query_key(residual_test[index]["input"])
        if tap_key != residual_key:
            query_mismatches += 1
        users.append(tap_key[0])

        tap_gold = tap_test[index]["output"]
        residual_gold = residual_test[index]["output"]
        if tap_predictions[index]["gold"] != tap_gold:
            prediction_gold_mismatches += 1
        if residual_predictions[index]["gold"] != residual_gold:
            prediction_gold_mismatches += 1

        tap_pid = tap_sid_to_pid.get(tap_gold)
        residual_pid = residual_sid_to_pid.get(residual_gold)
        if tap_pid is None or residual_pid is None:
            raise ValueError(f"Gold SID 无法映射到 PID：index={index}")
        if (
            tap_pid != residual_pid
            or tap_pid != test_targets[index]
        ):
            target_mismatches += 1
        target_pids.append(tap_pid)
        tap_metric_rows.append(
            rank_metrics(tap_predictions[index], "TAP-SID", index)
        )
        residual_metric_rows.append(
            rank_metrics(
                residual_predictions[index],
                "Residual SID",
                index,
            )
        )

    if query_mismatches or target_mismatches or prediction_gold_mismatches:
        raise ValueError(
            "严格匹配校验失败："
            f"query={query_mismatches}, "
            f"target={target_mismatches}, "
            f"prediction_gold={prediction_gold_mismatches}"
        )

    tap_metrics = np.vstack(tap_metric_rows)
    residual_metrics = np.vstack(residual_metric_rows)
    aggregate_tap = tap_metrics.mean(axis=0)
    aggregate_residual = residual_metrics.mean(axis=0)
    expected_tap = official_metrics(paths["tap_metrics"])
    expected_residual = official_metrics(paths["residual_metrics"])
    if not np.allclose(aggregate_tap, expected_tap, atol=1e-12):
        raise ValueError(
            "TAP-SID 总体指标与正式 metrics 不一致："
            f"recomputed={aggregate_tap.tolist()}, "
            f"official={expected_tap.tolist()}"
        )
    if not np.allclose(
        aggregate_residual,
        expected_residual,
        atol=1e-12,
    ):
        raise ValueError(
            "Residual SID 总体指标与正式 metrics 不一致："
            f"recomputed={aggregate_residual.tolist()}, "
            f"official={expected_residual.tolist()}"
        )

    boundaries = load_city_boundaries(
        paths["province_geojson"],
        args.city_boundary_dir,
    )
    pid_to_city: dict[str, str] = {}
    for pid in sorted(set(target_pids), key=int):
        if pid not in coordinates:
            raise ValueError(f"目标 PID 缺少坐标：{pid}")
        longitude, latitude = coordinates[pid]
        city, _ = assign_city(longitude, latitude, boundaries)
        pid_to_city[pid] = city

    cities = [pid_to_city[pid] for pid in target_pids]
    city_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, city in enumerate(cities):
        city_to_indices[city].append(index)

    rng = np.random.default_rng(args.seed)
    records = [
        metric_record(
            "总体",
            list(range(sample_count)),
            tap_metrics,
            residual_metrics,
            users,
            target_pids,
            args.bootstrap,
            rng,
        )
    ]
    for city in sorted(
        city_to_indices,
        key=lambda name: (-len(city_to_indices[name]), name),
    ):
        records.append(
            metric_record(
                city,
                city_to_indices[city],
                tap_metrics,
                residual_metrics,
                users,
                target_pids,
                args.bootstrap,
                rng,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / "city_r5_metrics_all.csv"
    output_json = args.output_dir / "city_r5_metrics_all.json"
    write_csv(output_csv, records)
    output_json.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    unmatched_samples = len(city_to_indices.get("未匹配", []))
    validation = {
        "status": "PASS",
        "samples": sample_count,
        "query_mismatches": query_mismatches,
        "target_pid_mismatches": target_mismatches,
        "prediction_gold_mismatches": prediction_gold_mismatches,
        "unique_target_pois": len(set(target_pids)),
        "unique_cities": len(city_to_indices),
        "unmatched_samples": unmatched_samples,
        "unmatched_ratio": unmatched_samples / sample_count,
        "boundary_features": len(boundaries),
        "aggregate_reproduced": {
            "residual": dict(
                zip(METRIC_NAMES, aggregate_residual.tolist())
            ),
            "tap": dict(zip(METRIC_NAMES, aggregate_tap.tolist())),
        },
        "bootstrap": {
            "iterations": args.bootstrap,
            "unit": "user within each city",
            "method": "paired percentile",
            "seed": args.seed,
        },
        "inputs": {
            label: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for label, path in paths.items()
        },
    }
    validation_path = args.output_dir / "validation_report.json"
    validation_path.write_text(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(validation, ensure_ascii=False, indent=2))
    print(f"CSV: {output_csv}")
    print(f"JSON: {output_json}")
    print(f"Validation: {validation_path}")


if __name__ == "__main__":
    main()
