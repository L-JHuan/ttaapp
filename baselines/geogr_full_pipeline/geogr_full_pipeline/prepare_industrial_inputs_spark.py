"""从 TAP 的共享 Spark 产物导出 GeoGR 所需的紧凑输入视图。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SharedSparkLayout:
    catalog: Path
    category_l1: Path
    category_l2: Path
    train_sequences: Path


def resolve_shared_layout(processed_root: Path) -> SharedSparkLayout:
    """验证并返回 TAP 已完成的共享 Spark 目录。"""
    layout = SharedSparkLayout(
        catalog=processed_root / "metadata" / "catalog",
        category_l1=processed_root / "mappings" / "category_l1",
        category_l2=processed_root / "mappings" / "category_l2",
        train_sequences=processed_root / "sequence_parquet" / "train",
    )
    missing = [path for path in layout.__dict__.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "缺少 TAP 共享 Spark 产物: " + ", ".join(str(path) for path in missing)
        )
    return layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize compact GeoGR inputs from TAP Spark outputs."
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shuffle_partitions", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _replace_file(temporary: Path, target: Path) -> None:
    if target.exists():
        target.unlink()
    temporary.replace(target)


def main() -> None:
    args = parse_args()
    layout = resolve_shared_layout(args.processed_root)
    report_path = args.output_dir / "industrial_input_report.json"
    outputs = {
        "poi_info": args.output_dir / "poi_info.csv",
        "role_priors": args.output_dir / "role_priors.csv",
        "train_sequences": args.output_dir / "train_poi_sequence.csv",
    }
    if report_path.exists() and all(path.exists() for path in outputs.values()):
        if not args.overwrite:
            print(report_path.read_text(encoding="utf-8"))
            return
    elif args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{args.output_dir} 已存在不完整产物；请使用新的 RUN_ROOT 或显式设置 --overwrite"
        )

    from pyspark.sql import SparkSession

    spark = SparkSession.builder.appName("GeoGR industrial input adapter").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        name: path.with_suffix(path.suffix + ".tmp") for name, path in outputs.items()
    }

    try:
        catalog = spark.read.parquet(str(layout.catalog))
        l1_mapping = spark.read.parquet(str(layout.category_l1))
        l2_mapping = spark.read.parquet(str(layout.category_l2))
        required_catalog = {"pid", "latitude", "longitude", "l1_label", "l2_label"}
        if missing := sorted(required_catalog - set(catalog.columns)):
            raise ValueError(f"metadata/catalog 缺少字段: {missing}")
        if not {"category_l1", "l1_label"}.issubset(l1_mapping.columns):
            raise ValueError("mappings/category_l1 缺少 category_l1 或 l1_label")
        if not {"category_l2", "l2_label"}.issubset(l2_mapping.columns):
            raise ValueError("mappings/category_l2 缺少 category_l2 或 l2_label")

        enriched = (
            catalog.join(l1_mapping.select("l1_label", "category_l1"), "l1_label", "inner")
            .join(l2_mapping.select("l2_label", "category_l2"), "l2_label", "inner")
            .select(
                "pid",
                "latitude",
                "longitude",
                "l1_label",
                "l2_label",
                "category_l1",
                "category_l2",
            )
            .orderBy("pid")
        )
        catalog_count = enriched.count()
        if catalog_count != catalog.count():
            raise ValueError("类别映射与 TAP 目录覆盖范围不一致")

        with temporary_paths["poi_info"].open(
            "w", encoding="utf-8", newline=""
        ) as poi_handle, temporary_paths["role_priors"].open(
            "w", encoding="utf-8", newline=""
        ) as role_handle:
            poi_writer = csv.writer(poi_handle)
            role_writer = csv.writer(role_handle)
            poi_writer.writerow(["pid", "latitude", "longitude"])
            role_writer.writerow(
                ["pid", "l1_label", "l2_label", "category_l1", "category_l2"]
            )
            for row in enriched.toLocalIterator(prefetchPartitions=True):
                pid = int(row["pid"])
                poi_writer.writerow([pid, float(row["latitude"]), float(row["longitude"])])
                role_writer.writerow(
                    [
                        pid,
                        int(row["l1_label"]),
                        int(row["l2_label"]),
                        str(row["category_l1"]),
                        str(row["category_l2"]),
                    ]
                )

        sequences = spark.read.parquet(str(layout.train_sequences))
        required_sequences = {
            "UserId",
            "target_utc",
            "sequence_PoiId",
            "sequence_UTCTimeOffset",
        }
        if missing := sorted(required_sequences - set(sequences.columns)):
            raise ValueError(f"sequence_parquet/train 缺少字段: {missing}")
        ordered_sequences = sequences.select(*sorted(required_sequences)).orderBy(
            "UserId", "target_utc"
        )
        sequence_count = ordered_sequences.count()
        with temporary_paths["train_sequences"].open(
            "w", encoding="utf-8", newline=""
        ) as sequence_handle:
            writer = csv.writer(sequence_handle)
            writer.writerow(
                ["UserId", "target_utc", "sequence_PoiId", "sequence_UTCTimeOffset"]
            )
            for row in ordered_sequences.toLocalIterator(prefetchPartitions=True):
                writer.writerow(
                    [
                        int(row["UserId"]),
                        str(row["target_utc"]),
                        json.dumps([int(value) for value in row["sequence_PoiId"]]),
                        json.dumps(
                            [str(value) for value in row["sequence_UTCTimeOffset"]],
                            ensure_ascii=False,
                        ),
                    ]
                )

        for name, target in outputs.items():
            _replace_file(temporary_paths[name], target)
        report = {
            "status": "GEOGR_INDUSTRIAL_SHARED_INPUTS_OK",
            "data_protocol": "reuse TAP Spark catalog, mappings, and train sequences",
            "processed_root": str(args.processed_root),
            "catalog_pois": int(catalog_count),
            "train_sequence_rows": int(sequence_count),
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        for path in temporary_paths.values():
            if path.exists():
                path.unlink()
        spark.stop()


if __name__ == "__main__":
    main()
