"""汇总 TAP-SID 与地理层级基线的同协议测试指标。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = ("recall@1", "recall@5", "recall@10", "ndcg@10")
ALIASES = {
    "recall@1": ("recall@1", "R@1", "r@1"),
    "recall@5": ("recall@5", "R@5", "r@5"),
    "recall@10": ("recall@10", "R@10", "r@10"),
    "ndcg@10": ("ndcg@10", "NDCG@10", "n@10"),
}


def _metric(payload: dict[str, Any], name: str) -> float:
    for key in ALIASES[name]:
        if key in payload:
            return float(payload[key])
    raise ValueError(f"metrics 缺少 {name}，可接受字段={ALIASES[name]}")


def _samples(payload: dict[str, Any]) -> int:
    for key in ("samples", "num_samples", "prediction_samples"):
        if key in payload:
            return int(payload[key])
    raise ValueError("metrics 缺少样本数字段")


def build_comparison(
    result_paths: dict[str, Path],
    reference: str,
) -> list[dict[str, Any]]:
    if reference not in result_paths:
        raise ValueError(f"reference={reference} 不在方法列表中")
    payloads = {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in result_paths.items()
    }
    sample_counts = {method: _samples(payload) for method, payload in payloads.items()}
    if len(set(sample_counts.values())) != 1:
        raise ValueError(f"各方法测试样本数不一致: {sample_counts}")

    reference_metrics = {
        metric: _metric(payloads[reference], metric) for metric in METRICS
    }
    rows: list[dict[str, Any]] = []
    for method, path in result_paths.items():
        row: dict[str, Any] = {
            "method": method,
            "metrics_path": str(path),
            "samples": sample_counts[method],
            "reference": reference,
        }
        for metric in METRICS:
            value = _metric(payloads[method], metric)
            reference_value = reference_metrics[metric]
            delta = value - reference_value
            relative = 0.0 if reference_value == 0 else delta / reference_value * 100.0
            row[metric] = value
            row[f"delta_{metric}"] = delta
            row[f"relative_{metric}_pct"] = relative
        rows.append(row)
    return rows


def _parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result 必须为 METHOD=/path/to/metrics.json")
    method, path = value.split("=", 1)
    if not method.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--result 的方法名和路径不能为空")
    return method.strip(), Path(path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare matched geographic SID baselines.")
    parser.add_argument(
        "--result",
        action="append",
        type=_parse_result,
        required=True,
        help="METHOD=/path/to/test_metrics.json; repeat for each method",
    )
    parser.add_argument("--reference", default="TAP-SID")
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = dict(args.result)
    if len(result_paths) != len(args.result):
        raise ValueError("--result 包含重复方法名")
    rows = build_comparison(result_paths, args.reference)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "status": "MATCHED_GEO_HIERARCHICAL_COMPARISON_OK",
                "reference": args.reference,
                "delta_definition": "method - reference",
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
