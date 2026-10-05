"""合并 GeoGR EM beam 候选分片，并检查目录覆盖与重复。"""

from __future__ import annotations

import argparse
from pathlib import Path

from geogr_full_pipeline.em_refine_sid import load_candidate_json, load_codebook
from geogr_full_pipeline.common import write_json


def merge_candidate_shards(
    shard_paths: list[Path], expected_pids: set[int]
) -> dict[int, list]:
    merged: dict[int, list] = {}
    for path in shard_paths:
        shard = load_candidate_json(path)
        overlap = sorted(set(merged) & set(shard))
        if overlap:
            raise ValueError(f"候选分片 POI 重复: {overlap[:10]}")
        merged.update(shard)
    if set(merged) != expected_pids:
        missing = sorted(expected_pids - set(merged))
        extra = sorted(set(merged) - expected_pids)
        raise ValueError(
            f"候选分片未完整覆盖目录: missing={missing[:10]}, extra={extra[:10]}"
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge GeoGR EM candidate shards")
    parser.add_argument("--input_sid_csv", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_pids = set(load_codebook(args.input_sid_csv))
    merged = merge_candidate_shards(args.shard, expected_pids)
    write_json(
        args.output_json,
        {
            "status": "GEOGR_FULL_EM_CANDIDATES_MERGED_OK",
            "shards": [str(path) for path in args.shard],
            "samples": len(merged),
            "candidates": {
                str(pid): [
                    {"code": list(row.code), "score": row.score, "rank": row.rank}
                    for row in rows
                ]
                for pid, rows in sorted(merged.items())
            },
        },
    )


if __name__ == "__main__":
    main()
