"""Build sharded TAP-SID SFT JSONL with Spark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


INSTRUCTION = (
    "Here is a record of a user's POI accesses, your task is based on the history "
    "to predict the POI that the user is likely to access at the specified time."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sharded TAP-SID SFT data with Spark.")
    parser.add_argument("--sid_csv", type=Path, required=True)
    parser.add_argument("--sequence_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--shuffle_partitions", type=int, default=2000)
    parser.add_argument("--output_partitions", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = SparkSession.builder.appName("TAP-SID Spark SFT builder").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    spark.conf.set("spark.sql.adaptive.enabled", "true")
    mode = "overwrite" if args.overwrite else "errorifexists"

    codebook = (
        spark.read.option("header", True)
        .csv(str(args.sid_csv))
        .select(F.col("pid").cast("long").alias("pid"), F.col("sid_tokens").alias("sid"))
    )
    if codebook.filter(F.col("pid").isNull() | F.col("sid").isNull()).limit(1).count():
        raise ValueError("TAP-SID codebook contains invalid pid or sid_tokens")

    splits = ["train"]
    if (args.sequence_root / "val").exists():
        splits.append("val")
    splits.append("test")

    reports: dict[str, dict[str, int]] = {}
    for split in splits:
        sequences = spark.read.parquet(str(args.sequence_root / split))
        exploded = sequences.select(
            "sample_id",
            "UserId",
            "target_utc",
            F.posexplode(
                F.arrays_zip("sequence_PoiId", "sequence_UTCTimeOffset")
            ).alias("position", "event"),
        ).select(
            "sample_id",
            "UserId",
            "target_utc",
            "position",
            F.col("event.sequence_PoiId").cast("long").alias("pid"),
            F.col("event.sequence_UTCTimeOffset").alias("event_time"),
        )
        joined = exploded.join(codebook, "pid", "left")
        missing = joined.filter(F.col("sid").isNull()).limit(1).count()
        if missing:
            raise ValueError(f"{split} contains a pid missing from the TAP-SID codebook")
        grouped = joined.groupBy("sample_id", "UserId", "target_utc").agg(
            F.sort_array(
                F.collect_list(F.struct("position", "event_time", "sid"))
            ).alias("events")
        )
        history = F.slice("events", 1, F.size("events") - 1)
        target = F.element_at("events", -1)
        history_text = F.concat_ws(
            ", ",
            F.transform(
                history,
                lambda item: F.concat(
                    item["event_time"],
                    F.lit(" visited "),
                    item["sid"],
                ),
            ),
        )
        result = grouped.select(
            F.lit(INSTRUCTION).alias("instruction"),
            F.concat(
                F.lit("User_"),
                F.col("UserId").cast("string"),
                F.lit(" checkin history: "),
                history_text,
                F.lit(".\nWhen "),
                target["event_time"],
                F.lit(" user_"),
                F.col("UserId").cast("string"),
                F.lit(" is likely to visit:"),
            ).alias("input"),
            target["sid"].alias("output"),
        )
        output_path = args.output_dir / f"llm_{split}.jsonl"
        (
            result.repartition(args.output_partitions)
            .write.mode(mode)
            .json(str(output_path))
        )
        reports[split] = {
            "samples": spark.read.json(str(output_path)).count(),
            "output_partitions": args.output_partitions,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "engine": "spark",
        "sid_csv": str(args.sid_csv),
        "sequence_root": str(args.sequence_root),
        "splits": reports,
        "format": "Spark JSON Lines directory",
    }
    (args.output_dir / "llm_json_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
