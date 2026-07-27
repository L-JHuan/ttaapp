"""从统一 Spark 预处理产物构造 GNPR 的 79 维 POI 表征。"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from gnpr_baseline.build_poi_embeddings import (
    category_features,
    spatial_features,
    temporal_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GNPR POI embeddings from TAP Spark outputs.")
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--train_end", required=True)
    parser.add_argument("--category_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--timezone_offset_minutes", type=int, default=480)
    parser.add_argument("--category_dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shuffle_partitions", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    spark = SparkSession.builder.appName("GNPR POI embedding builder").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = args.processed_root / "metadata" / "catalog"
    states_path = args.processed_root / "_spark_stages" / "catalog_filtered_states"
    poi_map_path = args.processed_root / "mappings" / "pois"
    for path in (catalog_path, states_path, poi_map_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少统一 Spark 预处理产物: {path}")

    catalog = spark.read.parquet(str(catalog_path))
    states = spark.read.parquet(str(states_path))
    poi_map = spark.read.parquet(str(poi_map_path)).select("_poi", "pid")
    catalog_columns = set(catalog.columns)
    category_source = "metadata/catalog"
    if "category_l2" not in catalog_columns:
        category_source = "_spark_stages/catalog_filtered_states"
        category_counts = states.groupBy("_poi", "category_l2").count()
        order = Window.partitionBy("_poi").orderBy(F.desc("count"), "category_l2")
        selected = (
            category_counts.withColumn("_rank", F.row_number().over(order))
            .filter(F.col("_rank") == 1)
            .select("_poi", "category_l2")
        )
        catalog = catalog.join(poi_map, "pid", "inner").join(selected, "_poi", "inner")

    catalog_rows = (
        catalog.select("pid", "latitude", "longitude", "category_l2")
        .orderBy("pid")
        .collect()
    )
    if not catalog_rows:
        raise ValueError("Spark POI 目录为空")
    categories = sorted({str(row["category_l2"]) for row in catalog_rows})
    category_to_embedding, effective_dim = category_features(
        categories,
        args.category_model,
        args.category_dim,
        args.device,
    )

    train_end = pd.Timestamp(args.train_end)
    if train_end.tzinfo is None:
        raise ValueError("train_end 必须包含时区")
    train_end_utc = train_end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    hour_rows = (
        states.filter(F.col("_time") <= F.to_timestamp(F.lit(train_end_utc)))
        .join(poi_map, "_poi", "inner")
        .withColumn(
            "local_hour",
            F.hour(
                F.col("_time")
                + F.expr(f"INTERVAL {int(args.timezone_offset_minutes)} MINUTES")
            ),
        )
        .groupBy("pid", "local_hour")
        .count()
        .collect()
    )
    hour_counts: dict[int, Counter[int]] = {}
    for row in hour_rows:
        hour_counts.setdefault(int(row["pid"]), Counter())[int(row["local_hour"])] = int(
            row["count"]
        )

    poi_vectors: dict[int, np.ndarray] = {}
    poi_info: list[dict[str, object]] = []
    for row in catalog_rows:
        pid = int(row["pid"])
        category = str(row["category_l2"])
        vector = np.concatenate(
            [
                category_to_embedding[category],
                spatial_features(float(row["latitude"]), float(row["longitude"])),
                temporal_features(hour_counts.get(pid, Counter())),
            ]
        ).astype(np.float32)
        poi_vectors[pid] = vector
        poi_info.append(
            {
                "pid": pid,
                "category": category,
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "visit_time_and_count": json.dumps(
                    dict(sorted(hour_counts.get(pid, Counter()).items()))
                ),
            }
        )

    with (args.output_dir / "poi_Emb_dict.pkl").open("wb") as handle:
        pickle.dump(poi_vectors, handle)
    with (args.output_dir / "category_to_embedding.pkl").open("wb") as handle:
        pickle.dump(category_to_embedding, handle)
    pd.DataFrame(poi_info).to_csv(args.output_dir / "poi_info_for_sid.csv", index=False)
    report = {
        "status": "GNPR_SPARK_POI_EMBEDDINGS_OK",
        "processed_root": str(args.processed_root),
        "train_end_utc": train_end.isoformat(),
        "pois": len(poi_vectors),
        "categories": len(categories),
        "category_dim": args.category_dim,
        "category_pca_effective_dim": effective_dim,
        "spatial_dim": 3,
        "temporal_dim": 12,
        "total_dim": len(next(iter(poi_vectors.values()))),
        "category_source": category_source,
        "data_protocol": "reuse TAP Spark mappings, catalog, states, and sequence splits",
    }
    (args.output_dir / "embedding_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
