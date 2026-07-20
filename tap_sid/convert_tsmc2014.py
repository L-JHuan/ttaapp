"""将 TSMC2014 八列签到文件转换为 TAP-SID 的事件表和 POI 表。"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import median
from typing import Any


def category_key(value: str) -> str:
    """生成稳定的 Foursquare 类别匹配键。"""
    text = value.strip().lower().replace("café", "cafe").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def load_category_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"无效的类别映射: {path}")
    return {category_key(str(name)): str(parent).strip() for name, parent in raw.items()}


def choose_mode(counter: Counter[tuple[str, str]]) -> tuple[str, str]:
    """按频次降序、类别 ID 和名称升序确定 POI 的细类别。"""
    return min(counter, key=lambda item: (-counter[item], item[0], item[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a raw TSMC2014 TSV file for TAP-SID.")
    parser.add_argument("--input", type=Path, required=True, help="TSMC2014 八列 TSV 文件。")
    parser.add_argument("--events_output", type=Path, required=True)
    parser.add_argument("--pois_output", type=Path, required=True)
    parser.add_argument("--report_output", type=Path, required=True)
    parser.add_argument("--category_l1_map", type=Path, required=True)
    parser.add_argument("--encoding", default="latin-1")
    return parser.parse_args()


def convert(
    input_path: Path,
    events_output: Path,
    pois_output: Path,
    report_output: Path,
    category_l1_map: Path,
    encoding: str,
) -> dict[str, Any]:
    l1_map = load_category_map(category_l1_map)
    events_output.parent.mkdir(parents=True, exist_ok=True)
    pois_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)

    user_counts: Counter[str] = Counter()
    user_pois: dict[str, set[str]] = defaultdict(set)
    poi_categories: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    poi_latitudes: dict[str, list[float]] = defaultdict(list)
    poi_longitudes: dict[str, list[float]] = defaultdict(list)
    poi_coordinate_values: dict[str, set[tuple[float, float]]] = defaultdict(set)
    rows = 0
    dropped_rows = 0

    with input_path.open("r", encoding=encoding, newline="") as source, events_output.open(
        "w", encoding="utf-8", newline=""
    ) as target:
        reader = csv.reader(source, delimiter="\t")
        writer = csv.DictWriter(
            target,
            fieldnames=["user_id", "poi_id", "timestamp", "timezone_offset_minutes"],
        )
        writer.writeheader()
        for raw in reader:
            if len(raw) != 8:
                dropped_rows += 1
                continue
            try:
                user_id = raw[0].strip()
                poi_id = raw[1].strip()
                category_id = raw[2].strip()
                category_name = raw[3].strip()
                latitude = float(raw[4])
                longitude = float(raw[5])
                timezone_offset = int(raw[6])
                timestamp = parsedate_to_datetime(raw[7]).astimezone(timezone.utc)
            except (TypeError, ValueError, IndexError, OverflowError):
                dropped_rows += 1
                continue
            if not user_id or not poi_id or not category_id or not category_name:
                dropped_rows += 1
                continue
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                dropped_rows += 1
                continue

            writer.writerow(
                {
                    "user_id": user_id,
                    "poi_id": poi_id,
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "timezone_offset_minutes": timezone_offset,
                }
            )
            rows += 1
            user_counts[user_id] += 1
            user_pois[user_id].add(poi_id)
            poi_categories[poi_id][(category_id, category_name)] += 1
            poi_latitudes[poi_id].append(latitude)
            poi_longitudes[poi_id].append(longitude)
            poi_coordinate_values[poi_id].add((latitude, longitude))

    if not rows:
        raise ValueError(f"{input_path} 没有可用的八列签到记录")

    unmapped_categories: Counter[str] = Counter()
    l1_values: set[str] = set()
    l2_values: set[str] = set()
    with pois_output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "poi_id",
                "latitude",
                "longitude",
                "category_l1",
                "category_l2",
                "category_l2_id",
            ],
        )
        writer.writeheader()
        for poi_id in sorted(poi_categories):
            category_id, category_name = choose_mode(poi_categories[poi_id])
            category_l1 = l1_map.get(category_key(category_name), "Other")
            if category_l1 == "Other" and category_key(category_name) not in l1_map:
                unmapped_categories[category_name] += 1
            l1_values.add(category_l1)
            l2_values.add(category_name)
            writer.writerow(
                {
                    "poi_id": poi_id,
                    "latitude": median(poi_latitudes[poi_id]),
                    "longitude": median(poi_longitudes[poi_id]),
                    "category_l1": category_l1,
                    "category_l2": category_name,
                    "category_l2_id": category_id,
                }
            )

    if len(l1_values) < 2:
        raise ValueError(
            "粗类别映射后只有一个 L1 类别。请通过 --category_l1_map 提供适用于当前数据的细类到粗类映射。"
        )

    max_checkins = max(user_counts.values())
    max_users = sorted(user_id for user_id, count in user_counts.items() if count == max_checkins)
    report: dict[str, Any] = {
        "input": str(input_path),
        "encoding": encoding,
        "rows": rows,
        "dropped_rows": dropped_rows,
        "users": len(user_counts),
        "pois": len(poi_categories),
        "categories_l1": len(l1_values),
        "categories_l2": len(l2_values),
        "category_conflict_pois": sum(len(values) > 1 for values in poi_categories.values()),
        "coordinate_conflict_pois": sum(len(values) > 1 for values in poi_coordinate_values.values()),
        "unmapped_l2_categories": len(unmapped_categories),
        "unmapped_pois": sum(unmapped_categories.values()),
        "unmapped_poi_ratio": sum(unmapped_categories.values()) / len(poi_categories),
        "unmapped_category_examples": [
            {"category": name, "pois": count} for name, count in unmapped_categories.most_common(20)
        ],
        "max_user_checkins": max_checkins,
        "max_user_ids": max_users,
        "max_user_unique_pois": {user_id: len(user_pois[user_id]) for user_id in max_users},
        "poi_metadata_rule": {
            "category": "most frequent (category_id, category_name), deterministic tie break",
            "coordinates": "median latitude and median longitude across check-ins",
            "category_l1": "static Foursquare category-name taxonomy; fallback=Other",
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
        category_l1_map=args.category_l1_map,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
