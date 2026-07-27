"""Spark preprocessing for large industrial next-POI event logs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"时间边界必须包含时区: {value}")
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def resolve_split_boundaries(
    train_end_value: str,
    validation_end_value: str,
) -> tuple[datetime, datetime | None]:
    train_end = parse_utc(train_end_value)
    validation_end = (
        parse_utc(validation_end_value)
        if validation_end_value.strip()
        else None
    )
    if validation_end is not None and validation_end <= train_end:
        raise ValueError("validation_end 必须晚于 train_end")
    return train_end, validation_end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Spark to prepare large industrial JSONL logs without event-level CSV files."
    )
    parser.add_argument("--input", required=True, help="JSONL file, glob, or directory understood by Spark")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_end", required=True, help="Inclusive UTC cutoff")
    parser.add_argument(
        "--validation_end",
        default="",
        help="Inclusive UTC validation cutoff; empty means train/test only",
    )
    parser.add_argument("--timezone_offset_minutes", type=int, default=480)
    parser.add_argument("--max_sequence_length", type=int, default=50)
    parser.add_argument("--min_history_length", type=int, default=1)
    parser.add_argument("--keep_last_k_train", type=int, default=5)
    parser.add_argument("--shuffle_partitions", type=int, default=2000)
    parser.add_argument("--output_partitions", type=int, default=256)
    parser.add_argument("--mapping_partitions", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _dense_mapping(frame: Any, key: str, value: str, partitions: int) -> Any:
    """Build a deterministic dense ID mapping without a single-partition SQL window."""
    spark = frame.sparkSession
    indexed = (
        frame.select(key)
        .distinct()
        .rdd.map(lambda row: str(row[0]))
        .sortBy(lambda item: item, numPartitions=partitions)
        .zipWithIndex()
        .map(lambda item: (item[0], int(item[1])))
    )
    return spark.createDataFrame(indexed, schema=f"{key} string, {value} long")


def main() -> None:
    args = parse_args()
    if args.max_sequence_length < 2:
        raise ValueError("max_sequence_length 必须至少为 2")
    if args.keep_last_k_train <= 0:
        raise ValueError("keep_last_k_train 必须大于 0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} 已存在且非空；如需覆盖请设置 --overwrite")

    from pyspark.sql import SparkSession, Window
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, StructField, StructType

    spark = SparkSession.builder.appName("GNPR industrial preprocessing").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

    mode = "overwrite" if args.overwrite else "errorifexists"
    output = args.output_dir
    stage_root = output / "_spark_stages"
    normalized_path = stage_root / "normalized_events"
    deduplicated_path = stage_root / "deduplicated_events"
    states_path = stage_root / "catalog_filtered_states"

    schema = StructType(
        [
            StructField("user_id", StringType()),
            StructField("poiid", StringType()),
            StructField("new_key_type", StringType()),
            StructField("longitude", StringType()),
            StructField("latitude", StringType()),
            StructField("log_time", StringType()),
        ]
    )
    raw = spark.read.schema(schema).json(args.input)
    first_path = F.trim(F.element_at(F.split(F.col("new_key_type"), r"\|"), 1))
    category_levels = F.split(first_path, ";")
    parsed_time = F.to_timestamp(F.trim(F.col("log_time")))
    explicit_zone = F.trim(F.col("log_time")).rlike(r"(Z|[+-]\d{2}:?\d{2})$")
    offset_interval = F.expr(f"INTERVAL {args.timezone_offset_minutes} MINUTES")
    utc_time = F.when(explicit_zone, parsed_time).otherwise(parsed_time - offset_interval)

    normalized = (
        raw.select(
            F.trim(F.col("user_id")).alias("_user"),
            F.trim(F.col("poiid")).alias("_poi"),
            F.col("longitude").cast("double").alias("longitude"),
            F.col("latitude").cast("double").alias("latitude"),
            first_path.alias("_category_path"),
            category_levels.alias("_category_levels"),
            utc_time.alias("_time"),
        )
        .filter(F.col("_user").isNotNull() & (F.length("_user") > 0))
        .filter(F.col("_poi").isNotNull() & (F.length("_poi") > 0))
        .filter(F.col("_time").isNotNull())
        .filter(F.col("latitude").between(-90.0, 90.0))
        .filter(F.col("longitude").between(-180.0, 180.0))
        .filter(F.size("_category_levels") == 3)
        .filter(F.expr("aggregate(_category_levels, true, (ok, x) -> ok AND length(trim(x)) > 0)"))
        .select(
            "_user",
            "_poi",
            "_time",
            "latitude",
            "longitude",
            F.trim(F.element_at("_category_levels", 1)).alias("category_l1"),
            F.trim(F.element_at("_category_levels", 3)).alias("category_l2"),
            "_category_path",
        )
        .withColumn(
            "_user_bucket",
            F.pmod(F.xxhash64("_user"), F.lit(args.output_partitions)).cast("int"),
        )
    )
    (
        normalized.repartition(args.output_partitions, "_user_bucket")
        .write.mode(mode)
        .partitionBy("_user_bucket")
        .parquet(str(normalized_path))
    )
    normalized = spark.read.parquet(str(normalized_path))

    deduplicated = normalized.dropDuplicates(["_user", "_poi", "_time"])
    (
        deduplicated.repartition(args.output_partitions, "_user_bucket")
        .write.mode(mode)
        .partitionBy("_user_bucket")
        .parquet(str(deduplicated_path))
    )
    deduplicated = spark.read.parquet(str(deduplicated_path))

    train_end, validation_end = resolve_split_boundaries(
        args.train_end,
        args.validation_end,
    )
    train_events = deduplicated.filter(F.col("_time") <= F.lit(train_end))
    train_users = train_events.select("_user").distinct()
    train_pois = train_events.select("_poi").distinct()

    filtered = (
        deduplicated.join(train_users, "_user", "left_semi")
        .join(train_pois, "_poi", "left_semi")
    )
    user_order = Window.partitionBy("_user").orderBy(F.col("_time"), F.col("_poi"))
    states = (
        filtered.withColumn("_previous_poi", F.lag("_poi").over(user_order))
        .filter(F.col("_previous_poi").isNull() | (F.col("_poi") != F.col("_previous_poi")))
        .drop("_previous_poi")
    )
    (
        states.repartition(args.output_partitions, "_user_bucket")
        .write.mode(mode)
        .partitionBy("_user_bucket")
        .parquet(str(states_path))
    )
    states = spark.read.parquet(str(states_path))

    category_counts = (
        normalized.groupBy("_poi", "category_l1", "category_l2", "_category_path")
        .count()
        .join(train_pois, "_poi", "left_semi")
    )
    category_order = Window.partitionBy("_poi").orderBy(
        F.desc("count"), "category_l1", "category_l2", "_category_path"
    )
    selected_category = (
        category_counts.withColumn("_rank", F.row_number().over(category_order))
        .filter(F.col("_rank") == 1)
        .drop("_rank", "count")
    )
    coordinates = (
        normalized.join(train_pois, "_poi", "left_semi")
        .groupBy("_poi")
        .agg(
            F.expr("percentile_approx(latitude, 0.5, 10000)").alias("latitude"),
            F.expr("percentile_approx(longitude, 0.5, 10000)").alias("longitude"),
        )
    )
    poi_metadata = coordinates.join(selected_category, "_poi", "inner")

    user_map = _dense_mapping(states, "_user", "UserId", args.mapping_partitions)
    poi_map = _dense_mapping(poi_metadata, "_poi", "pid", args.mapping_partitions)
    l1_map = _dense_mapping(poi_metadata, "category_l1", "l1_label", 1)
    l2_map = _dense_mapping(poi_metadata, "category_l2", "l2_label", 1)

    l2_conflicts = (
        poi_metadata.groupBy("category_l2")
        .agg(F.countDistinct("category_l1").alias("parents"))
        .filter(F.col("parents") > 1)
        .limit(10)
        .collect()
    )
    if l2_conflicts:
        examples = [row["category_l2"] for row in l2_conflicts]
        raise ValueError(f"同一 L2 类别对应多个 L1 类别: {examples}")

    mappings_root = output / "mappings"
    user_map.write.mode(mode).parquet(str(mappings_root / "users"))
    poi_map.write.mode(mode).parquet(str(mappings_root / "pois"))
    l1_map.write.mode(mode).parquet(str(mappings_root / "category_l1"))
    l2_map.write.mode(mode).parquet(str(mappings_root / "category_l2"))

    catalog = (
        poi_metadata.join(poi_map, "_poi", "inner")
        .join(l1_map, "category_l1", "inner")
        .join(l2_map, "category_l2", "inner")
    )
    metadata_root = output / "metadata"
    (
        catalog.select(
            "pid",
            "latitude",
            "longitude",
            "category_l1",
            "category_l2",
            "_category_path",
            "l1_label",
            "l2_label",
        )
        .repartition(min(args.output_partitions, 64), "pid")
        .write.mode(mode)
        .parquet(str(metadata_root / "catalog"))
    )

    local_time = F.date_format(
        F.col("_time") + offset_interval,
        "yyyy-MM-dd HH:mm:ss",
    )
    indexed_states = (
        states.join(user_map, "_user", "inner")
        .join(poi_map, "_poi", "inner")
        .withColumn("_local_time", local_time)
    )
    sequence_order = Window.partitionBy("_user").orderBy(F.col("_time"), F.col("_poi"))
    sequence_window = sequence_order.rowsBetween(-(args.max_sequence_length - 1), 0)
    samples = (
        indexed_states.withColumn("_user_position", F.row_number().over(sequence_order))
        .withColumn(
            "_sequence",
            F.collect_list(
                F.struct(
                    F.col("_time").alias("event_time"),
                    F.col("_poi").alias("event_poi"),
                    F.col("pid").cast("long").alias("pid"),
                    F.col("_local_time").alias("local_time"),
                )
            ).over(sequence_window),
        )
        .filter(F.col("_user_position") > args.min_history_length)
        .withColumn("target_utc", F.date_format("_time", "yyyy-MM-dd'T'HH:mm:ss'Z'"))
        .withColumn("sequence_PoiId", F.transform("_sequence", lambda item: item["pid"]))
        .withColumn(
            "sequence_UTCTimeOffset",
            F.transform("_sequence", lambda item: item["local_time"]),
        )
        .withColumn(
            "sample_id",
            F.sha2(
                F.to_json(
                    F.struct("UserId", "target_utc", "sequence_PoiId")
                ),
                256,
            ),
        )
        .select(
            "sample_id",
            "UserId",
            "target_utc",
            "sequence_PoiId",
            "sequence_UTCTimeOffset",
            "_time",
            "_poi",
            "_user",
        )
    )
    train_samples = samples.filter(F.col("_time") <= F.lit(train_end))
    train_rank = Window.partitionBy("_user").orderBy(F.desc("_time"), F.desc("_poi"))
    train_samples = (
        train_samples.withColumn("_train_rank", F.row_number().over(train_rank))
        .filter(F.col("_train_rank") <= args.keep_last_k_train)
        .drop("_train_rank")
    )
    if validation_end is not None:
        validation_samples = samples.filter(
            (F.col("_time") > F.lit(train_end))
            & (F.col("_time") <= F.lit(validation_end))
        )
        test_samples = samples.filter(F.col("_time") > F.lit(validation_end))
    else:
        validation_samples = None
        test_samples = samples.filter(F.col("_time") > F.lit(train_end))

    sequence_root = output / "sequence_parquet"
    output_columns = [
        "sample_id",
        "UserId",
        "target_utc",
        "sequence_PoiId",
        "sequence_UTCTimeOffset",
    ]
    (
        train_samples.select(*output_columns)
        .repartition(args.output_partitions, "UserId")
        .write.mode(mode)
        .parquet(str(sequence_root / "train"))
    )
    if validation_samples is not None:
        (
            validation_samples.select(*output_columns)
            .repartition(args.output_partitions, "UserId")
            .write.mode(mode)
            .parquet(str(sequence_root / "val"))
        )
    (
        test_samples.select(*output_columns)
        .repartition(args.output_partitions, "UserId")
        .write.mode(mode)
        .parquet(str(sequence_root / "test"))
    )

    counts = {
        "normalized_events": normalized.count(),
        "deduplicated_events": deduplicated.count(),
        "catalog_filtered_states": states.count(),
        "users": user_map.count(),
        "catalog_pois": poi_map.count(),
        "categories_l1": l1_map.count(),
        "categories_l2": l2_map.count(),
        "train_samples_last_k": spark.read.parquet(str(sequence_root / "train")).count(),
        "test_samples": spark.read.parquet(str(sequence_root / "test")).count(),
    }
    if validation_samples is not None:
        counts["validation_samples"] = spark.read.parquet(
            str(sequence_root / "val")
        ).count()
    report = {
        "engine": "spark",
        "input": args.input,
        "train_end_utc": args.train_end,
        "validation_end_utc": args.validation_end or None,
        "timezone_offset_minutes": args.timezone_offset_minutes,
        "max_sequence_length": args.max_sequence_length,
        "min_history_length": args.min_history_length,
        "keep_last_k_train": args.keep_last_k_train,
        "shuffle_partitions": args.shuffle_partitions,
        "output_partitions": args.output_partitions,
        "counts": counts,
        "storage": {
            "event_level_csv_written": False,
            "sequence_csv_written": False,
            "intermediate_format": "partitioned parquet",
            "poi_level_metadata": "partitioned parquet",
            "final_codebook": "one row per POI; written by the GNPR residual codebook builder",
        },
        "assumptions": {
            "naive_log_time": f"UTC{args.timezone_offset_minutes / 60:+g}",
            "category": "first pipe-separated path; L1=first level, L2=third level",
            "poi_coordinates": "Spark percentile_approx median with accuracy=10000",
            "catalog": "train-seen users and POIs only",
            "test_history": "rolling history without future events",
        },
    }
    report_path = output / "spark_protocol_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
