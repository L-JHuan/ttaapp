"""校验 GNPR codebook 与统一 POI 目录是否严格对齐。"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd


SID_PATTERN = re.compile(r"<[a-z]_\d+>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid_csv", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--spark_catalog", type=Path)
    return parser.parse_args()


def expected_pids(args: argparse.Namespace) -> set[int]:
    if bool(args.id_mappings) == bool(args.spark_catalog):
        raise ValueError("必须且只能提供 id_mappings 或 spark_catalog")
    if args.id_mappings:
        mappings = json.loads(args.id_mappings.read_text(encoding="utf-8"))
        return {int(value) for value in mappings["poi_id_to_internal"].values()}
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("GNPR codebook validator").getOrCreate()
    values = {
        int(row["pid"])
        for row in spark.read.parquet(str(args.spark_catalog)).select("pid").collect()
    }
    spark.stop()
    return values


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.sid_csv)
    required = {"pid", "sid", "sid_tokens"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"GNPR codebook 缺少字段: {missing}")
    if frame["pid"].duplicated().any():
        raise ValueError("GNPR codebook 包含重复 pid")
    actual = {int(value) for value in frame["pid"]}
    expected = expected_pids(args)
    if actual != expected:
        raise ValueError(
            f"GNPR codebook 与目录不一致: missing={len(expected-actual)}, extra={len(actual-expected)}"
        )

    paths: list[tuple[int, ...]] = []
    for row in frame.itertuples(index=False):
        code = tuple(int(value) for value in ast.literal_eval(str(row.sid)))
        tokens = SID_PATTERN.findall(str(row.sid_tokens))
        if len(code) not in (3, 4) or len(tokens) != len(code):
            raise ValueError(f"pid={row.pid} 的 SID 层数或 token 格式错误")
        paths.append(code)
    if len(set(paths)) != len(paths):
        raise ValueError("添加 collision leaf 后仍存在重复完整 SID")
    print(
        json.dumps(
            {
                "status": "GNPR_CODEBOOK_VALID",
                "pois": len(paths),
                "three_level": sum(len(path) == 3 for path in paths),
                "four_level": sum(len(path) == 4 for path in paths),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
