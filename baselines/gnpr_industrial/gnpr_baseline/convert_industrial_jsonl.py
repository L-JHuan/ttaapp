"""将工业行为 JSONL 转换为 GNPR 基线使用的事件表和 POI 表。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any


REQUIRED_FIELDS = ("user_id", "poiid", "new_key_type", "longitude", "latitude", "log_time")


def parse_category_path(value: str) -> tuple[str, str, str, int]:
    """选择第一条三级类别路径，并返回 L1、L2、完整路径和路径数量。"""
    paths = [part.strip() for part in value.split("|") if part.strip()]
    if not paths:
        raise ValueError("类别路径为空")
    levels = [part.strip() for part in paths[0].split(";")]
    if len(levels) != 3 or any(not part for part in levels):
        raise ValueError(f"主类别路径不是三级结构: {paths[0]}")
    return levels[0], levels[2], paths[0], len(paths)


def parse_time(value: str, offset_minutes: int) -> datetime:
    """将日志时间转换为带时区的 UTC 时间。"""
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone(timedelta(minutes=offset_minutes)))
    return parsed.astimezone(timezone.utc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert industrial behavior JSONL for GNPR.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--events_output", type=Path, required=True)
    parser.add_argument("--pois_output", type=Path, required=True)
    parser.add_argument("--report_output", type=Path, required=True)
    parser.add_argument("--timezone_offset_minutes", type=int, default=480)
    return parser.parse_args()


def convert(
    input_path: Path,
    events_output: Path,
    pois_output: Path,
    report_output: Path,
    timezone_offset_minutes: int,
) -> dict[str, Any]:
    events_output.parent.mkdir(parents=True, exist_ok=True)
    pois_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    user_counts: Counter[str] = Counter()
    poi_counts: Counter[str] = Counter()
    poi_categories: dict[str, Counter[tuple[str, str, str]]] = defaultdict(Counter)
    poi_latitudes: dict[str, list[float]] = defaultdict(list)
    poi_longitudes: dict[str, list[float]] = defaultdict(list)
    rows = 0
    dropped_rows = 0
    multi_category_rows = 0

    with input_path.open("r", encoding="utf-8-sig") as source, events_output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["user_id", "poi_id", "timestamp", "timezone_offset_minutes"],
        )
        writer.writeheader()
        for line in source:
            try:
                record = json.loads(line)
                if any(record.get(field) in (None, "") for field in REQUIRED_FIELDS):
                    raise ValueError("存在空字段")
                user_id = str(record["user_id"]).strip()
                poi_id = str(record["poiid"]).strip()
                longitude = float(record["longitude"])
                latitude = float(record["latitude"])
                if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                    raise ValueError("经纬度超出范围")
                category_l1, category_l2, primary_path, path_count = parse_category_path(
                    str(record["new_key_type"])
                )
                timestamp = parse_time(str(record["log_time"]), timezone_offset_minutes)
            except (json.JSONDecodeError, TypeError, ValueError, KeyError, OverflowError):
                dropped_rows += 1
                continue

            writer.writerow(
                {
                    "user_id": user_id,
                    "poi_id": poi_id,
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "timezone_offset_minutes": timezone_offset_minutes,
                }
            )
            rows += 1
            multi_category_rows += int(path_count > 1)
            user_counts[user_id] += 1
            poi_counts[poi_id] += 1
            poi_categories[poi_id][(category_l1, category_l2, primary_path)] += 1
            poi_latitudes[poi_id].append(latitude)
            poi_longitudes[poi_id].append(longitude)

    if not rows:
        raise ValueError(f"{input_path} 没有可用记录")

    category_conflict_pois = 0
    l1_values: set[str] = set()
    l2_values: set[str] = set()
    with pois_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["poi_id", "latitude", "longitude", "category_l1", "category_l2"],
        )
        writer.writeheader()
        for poi_id in sorted(poi_categories):
            counter = poi_categories[poi_id]
            if len(counter) > 1:
                category_conflict_pois += 1
            category_l1, category_l2, _ = min(
                counter,
                key=lambda item: (-counter[item], item[0], item[1], item[2]),
            )
            l1_values.add(category_l1)
            l2_values.add(category_l2)
            writer.writerow(
                {
                    "poi_id": poi_id,
                    "latitude": median(poi_latitudes[poi_id]),
                    "longitude": median(poi_longitudes[poi_id]),
                    "category_l1": category_l1,
                    "category_l2": category_l2,
                }
            )

    report: dict[str, Any] = {
        "input": str(input_path),
        "rows": rows,
        "dropped_rows": dropped_rows,
        "users": len(user_counts),
        "pois": len(poi_counts),
        "categories_l1": len(l1_values),
        "categories_l2": len(l2_values),
        "multi_category_rows": multi_category_rows,
        "category_conflict_pois": category_conflict_pois,
        "min_user_events": min(user_counts.values()),
        "max_user_events": max(user_counts.values()),
        "assumptions": {
            "log_time": f"naive timestamps interpreted as UTC{timezone_offset_minutes / 60:+g}",
            "coordinates": "input coordinates kept unchanged; coordinate reference system must be confirmed",
            "category": "first pipe-separated path; L1=first level, L2=third level",
            "poi_coordinates": "median latitude and longitude across events",
        },
    }
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    args = parse_args()
    convert(
        input_path=args.input,
        events_output=args.events_output,
        pois_output=args.pois_output,
        report_output=args.report_output,
        timezone_offset_minutes=args.timezone_offset_minutes,
    )


if __name__ == "__main__":
    main()
