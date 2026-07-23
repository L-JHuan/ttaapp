from __future__ import annotations

import argparse
import json
from pathlib import Path

from tap_sid.eval_sharding import atomic_write_json, merge_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="合并多 GPU TAP-SID 评估分片并重新计算全量指标。")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shard_dir", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_predictions", type=Path, required=True)
    parser.add_argument("--output_metrics", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    predictions, metrics = merge_shards(
        args.dataset,
        args.shard_dir,
        args.num_shards,
        k=args.k,
    )
    atomic_write_json(args.output_predictions, predictions)
    atomic_write_json(args.output_metrics, metrics)
    print(f"Saved merged predictions -> {args.output_predictions}")
    print(f"Saved merged metrics -> {args.output_metrics}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
