"""由 P2P 微调后的 POI 表示构建三层初始 RQ-Kmeans SID。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from geogr_full_pipeline.common import load_embeddings, write_json
from geogr_full_pipeline.sid_utils import residual_kmeans, sid_tokens


def write_initial_codebook(path: Path, pids: list[int], code_rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "sid", "sid_tokens"])
        writer.writeheader()
        for pid, values in zip(pids, code_rows):
            code = [int(value) for value in values]
            writer.writerow({"pid": pid, "sid": str(code), "sid_tokens": sid_tokens(code)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial three-token GeoGR RQ SID")
    parser.add_argument("--embeddings_npz", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pids, embeddings = load_embeddings(args.embeddings_npz)
    codes, _, quantization = residual_kmeans(
        embeddings,
        codebook_size=args.codebook_size,
        num_layers=3,
        seed=args.seed,
    )
    write_initial_codebook(args.output_csv, pids, codes)
    counts = Counter(tuple(int(value) for value in row) for row in codes)
    write_json(
        args.report_json,
        {
            "status": "GEOGR_FULL_INITIAL_RQ_OK",
            "catalog_pois": len(pids),
            "codebook_size": args.codebook_size,
            "num_layers": 3,
            "unique_paths": len(counts),
            "colliding_paths": sum(1 for count in counts.values() if count > 1),
            "colliding_pois": sum(count for count in counts.values() if count > 1),
            "max_collision_group": max(counts.values()),
            "collision_leaf_added": False,
            "quantization": quantization,
            "output_csv": str(args.output_csv),
        },
    )


if __name__ == "__main__":
    main()
