from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tap_sid.json_records import iter_json_records, load_json_records


SID_PATTERN = re.compile(r"<[a-z]_\d+>")
SAMPLE_INDEX_KEY = "_tap_sample_index"


def read_json_list(path: Path) -> list[dict[str, Any]]:
    return load_json_records(path)


def canonical_sid(value: Any) -> str:
    if isinstance(value, list):
        value = "".join(str(item) for item in value)
    return "".join(SID_PATTERN.findall(str(value or "")))


def split_rows(rows: list[dict[str, Any]], num_shards: int) -> list[list[dict[str, Any]]]:
    if num_shards <= 0:
        raise ValueError("num_shards 必须大于 0")
    if not rows:
        raise ValueError("测试集不能为空")
    if num_shards > len(rows):
        raise ValueError(f"GPU/分片数 {num_shards} 不能超过测试样本数 {len(rows)}")

    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    for sample_index, row in enumerate(rows):
        item = dict(row)
        item[SAMPLE_INDEX_KEY] = sample_index
        shards[sample_index % num_shards].append(item)
    return shards


def write_shards(dataset: Path, output_dir: Path, num_shards: int) -> dict[str, Any]:
    if num_shards <= 0:
        raise ValueError("num_shards 必须大于 0")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [output_dir / f"dataset_{index:05d}.json" for index in range(num_shards)]
    handles = [path.open("w", encoding="utf-8") for path in paths]
    shard_samples = [0] * num_shards
    samples = 0
    try:
        for sample_index, row in enumerate(iter_json_records(dataset)):
            shard_index = sample_index % num_shards
            item = dict(row)
            item[SAMPLE_INDEX_KEY] = sample_index
            handles[shard_index].write(json.dumps(item, ensure_ascii=False) + "\n")
            shard_samples[shard_index] += 1
            samples += 1
    finally:
        for handle in handles:
            handle.close()
    if samples == 0:
        raise ValueError("测试集不能为空")
    if num_shards > samples:
        raise ValueError(f"GPU/分片数 {num_shards} 不能超过测试样本数 {samples}")

    manifest = {
        "dataset": str(dataset),
        "samples": samples,
        "num_shards": num_shards,
        "shard_samples": shard_samples,
        "shard_files": [str(path) for path in paths],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def compute_metrics(
    dataset_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    k: int = 10,
) -> dict[str, Any]:
    if len(dataset_rows) != len(prediction_rows):
        raise ValueError(
            f"数据集与预测数量不一致: dataset={len(dataset_rows)}, predictions={len(prediction_rows)}"
        )

    hits = {1: 0, 5: 0, 10: 0}
    ndcg10 = 0.0
    unique_sum = 0
    for dataset_row, prediction_row in zip(dataset_rows, prediction_rows):
        gold = canonical_sid(dataset_row.get("output", dataset_row.get("gold", "")))
        predictions = prediction_row.get("predictions", [])
        if not isinstance(predictions, list):
            raise ValueError("predictions 字段必须是列表")

        seen: set[str] = set()
        unique: list[str] = []
        for prediction in predictions:
            sid = canonical_sid(prediction)
            if sid and sid not in seen:
                seen.add(sid)
                unique.append(sid)
            if len(unique) >= k:
                break
        unique_sum += len(unique)
        for cutoff in hits:
            if gold in unique[:cutoff]:
                hits[cutoff] += 1
        if gold in unique[:10]:
            rank = unique.index(gold) + 1
            ndcg10 += 1.0 / math.log2(rank + 1)

    denominator = max(len(dataset_rows), 1)
    return {
        "recall@1": hits[1] / denominator,
        "recall@5": hits[5] / denominator,
        "recall@10": hits[10] / denominator,
        "ndcg@10": ndcg10 / denominator,
        "avg_unique_predictions": unique_sum / denominator,
        "samples": len(dataset_rows),
    }


def merge_shards(
    dataset: Path,
    shard_dir: Path,
    num_shards: int,
    k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_rows = read_json_list(dataset)
    indexed_predictions: dict[int, dict[str, Any]] = {}
    illegal_predictions = 0
    length_histogram: Counter[str] = Counter()
    shard_samples: list[int] = []
    shared_metrics: dict[str, Any] = {}
    shared_metric_keys = (
        "constrained_sid",
        "num_beams",
        "k",
        "legal_sid_checked",
        "legal_sid_count",
    )

    for shard_index in range(num_shards):
        predictions_path = shard_dir / f"predictions_{shard_index:05d}.json"
        metrics_path = shard_dir / f"metrics_{shard_index:05d}.json"
        if not predictions_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"分片 {shard_index} 的预测或指标文件缺失")

        prediction_rows = read_json_list(predictions_path)
        shard_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(shard_metrics, dict):
            raise ValueError(f"{metrics_path} 必须是 JSON 对象")
        if int(shard_metrics.get("shard_index", -1)) != shard_index:
            raise ValueError(f"分片编号不一致: {metrics_path}")
        if int(shard_metrics.get("num_shards", -1)) != num_shards:
            raise ValueError(f"分片总数不一致: {metrics_path}")
        if int(shard_metrics.get("samples", -1)) != len(prediction_rows):
            raise ValueError(f"分片样本数不一致: {metrics_path}")
        for key in shared_metric_keys:
            if key not in shard_metrics:
                continue
            if key in shared_metrics and shared_metrics[key] != shard_metrics[key]:
                raise ValueError(f"分片配置 {key} 不一致: {metrics_path}")
            shared_metrics[key] = shard_metrics[key]

        shard_samples.append(len(prediction_rows))
        illegal_predictions += int(shard_metrics.get("illegal_sid_predictions", 0))
        for key, value in shard_metrics.get("pred_sid_length_histogram", {}).items():
            length_histogram[str(key)] += int(value)

        for row in prediction_rows:
            sample_index = row.get("sample_index")
            if not isinstance(sample_index, int):
                raise ValueError(f"{predictions_path} 中存在无效 sample_index")
            if sample_index in indexed_predictions:
                raise ValueError(f"样本 {sample_index} 在多个分片中重复出现")
            indexed_predictions[sample_index] = row

    expected_indices = set(range(len(dataset_rows)))
    actual_indices = set(indexed_predictions)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)[:10]
        extra = sorted(actual_indices - expected_indices)[:10]
        raise ValueError(f"分片覆盖不完整: missing={missing}, extra={extra}")

    indexed_rows = [indexed_predictions[index] for index in range(len(dataset_rows))]
    metric_out = compute_metrics(dataset_rows, indexed_rows, k=k)
    merged_predictions = [
        {key: value for key, value in row.items() if key != "sample_index"}
        for row in indexed_rows
    ]
    metric_out.update(
        {
            "prediction_samples": len(merged_predictions),
            "constrained_sid": True,
            "k": k,
            "multi_gpu_eval": num_shards > 1,
            "num_shards": num_shards,
            "shard_samples": shard_samples,
            "illegal_sid_predictions": illegal_predictions,
            "pred_sid_length_histogram": dict(length_histogram),
            "sample_coverage_checked": True,
            **shared_metrics,
        }
    )
    return merged_predictions, metric_out


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)
