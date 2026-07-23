"""将事件级真实世界数据转换为 TAP-SID 的时间切分序列协议。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def require_columns(df: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{source} 缺少字段: {missing}")


def stable_mapping(values: pd.Series) -> dict[str, int]:
    labels = sorted({str(value) for value in values.dropna()})
    return {label: index for index, label in enumerate(labels)}


def local_time_text(timestamp: pd.Timestamp, offset_minutes: int) -> str:
    return (timestamp + pd.Timedelta(minutes=offset_minutes)).strftime("%Y-%m-%d %H:%M:%S")


def collapse_consecutive_same_poi(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """合并每位用户连续上报的相同 POI，并保留该状态的首次上报。"""
    ordered = events.sort_values(["_user", "_time", "_poi"], kind="stable").copy()
    previous_poi = ordered.groupby("_user", sort=False)["_poi"].shift()
    keep = previous_poi.isna() | ordered["_poi"].ne(previous_poi)
    collapsed = ordered.loc[keep].copy()
    return collapsed, len(ordered) - len(collapsed)


def assign_split(
    target_time: pd.Timestamp,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp | None,
) -> str:
    """按全局时间边界分配目标；无验证集时训练边界后的目标全部进入测试集。"""
    if target_time <= train_end:
        return "train"
    if validation_end is not None and target_time <= validation_end:
        return "validation"
    return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chronological TAP-SID sequences from event logs.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--pois", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_end", required=True, help="UTC cutoff, inclusive, e.g. 2025-01-01T00:00:00Z")
    parser.add_argument("--validation_end", help="UTC cutoff, inclusive")
    parser.add_argument(
        "--no_validation",
        action="store_true",
        help="Use a train/test time split and omit the validation split.",
    )
    parser.add_argument("--user_col", default="user_id")
    parser.add_argument("--poi_col", default="poi_id")
    parser.add_argument("--timestamp_col", default="timestamp")
    parser.add_argument("--timezone_offset_col", default="timezone_offset_minutes")
    parser.add_argument("--default_timezone_offset_minutes", type=int)
    parser.add_argument("--latitude_col", default="latitude")
    parser.add_argument("--longitude_col", default="longitude")
    parser.add_argument("--category_l1_col", default="category_l1")
    parser.add_argument("--category_l2_col", default="category_l2")
    parser.add_argument("--city_col", default="city_id")
    parser.add_argument("--city", default="")
    parser.add_argument("--event_type_col", default="")
    parser.add_argument("--event_type", default="")
    parser.add_argument("--max_sequence_length", type=int, default=50)
    parser.add_argument("--min_history_length", type=int, default=1)
    parser.add_argument(
        "--collapse_consecutive_same_poi",
        action="store_true",
        help="Collapse consecutive reports at the same POI into one observed state.",
    )
    parser.add_argument(
        "--catalog_scope",
        choices=["train_seen", "provided"],
        default="train_seen",
        help="train_seen matches the closed-catalog paper protocol; provided permits metadata-only POIs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_sequence_length < 2:
        raise ValueError("max_sequence_length 必须至少为 2")

    events = pd.read_csv(args.events)
    pois = pd.read_csv(args.pois)
    require_columns(events, [args.user_col, args.poi_col, args.timestamp_col], args.events)
    require_columns(
        pois,
        [args.poi_col, args.latitude_col, args.longitude_col, args.category_l1_col, args.category_l2_col],
        args.pois,
    )

    if args.city:
        require_columns(events, [args.city_col], args.events)
        require_columns(pois, [args.city_col], args.pois)
        events = events[events[args.city_col].astype(str) == args.city].copy()
        pois = pois[pois[args.city_col].astype(str) == args.city].copy()
    if args.event_type:
        if not args.event_type_col:
            raise ValueError("指定 event_type 时必须提供 event_type_col")
        require_columns(events, [args.event_type_col], args.events)
        events = events[events[args.event_type_col].astype(str) == args.event_type].copy()

    events["_user"] = events[args.user_col].astype(str)
    events["_poi"] = events[args.poi_col].astype(str)
    pois["_poi"] = pois[args.poi_col].astype(str)
    if pois["_poi"].duplicated().any():
        raise ValueError("POI 元数据中 poi_id 不唯一")
    events["_time"] = pd.to_datetime(events[args.timestamp_col], utc=True, errors="coerce")
    if events["_time"].isna().any():
        raise ValueError(f"存在无法解析的时间戳: {int(events['_time'].isna().sum())} 条")

    if args.timezone_offset_col in events.columns:
        events["_offset"] = pd.to_numeric(events[args.timezone_offset_col], errors="coerce")
        if events["_offset"].isna().any():
            raise ValueError("timezone_offset_minutes 存在缺失或非数值")
    elif args.default_timezone_offset_minutes is not None:
        events["_offset"] = int(args.default_timezone_offset_minutes)
    else:
        raise ValueError("必须提供逐事件时区偏移列，或设置 default_timezone_offset_minutes")
    events["_offset"] = events["_offset"].astype(int)

    for column in (args.latitude_col, args.longitude_col):
        pois[column] = pd.to_numeric(pois[column], errors="coerce")
    if pois[[args.latitude_col, args.longitude_col]].isna().any(axis=None):
        raise ValueError("POI 经纬度存在缺失或非数值")
    if not pois[args.latitude_col].between(-90, 90).all() or not pois[args.longitude_col].between(-180, 180).all():
        raise ValueError("POI 经纬度超出 WGS84 数值范围；请先确认坐标系并转换")
    if pois[[args.category_l1_col, args.category_l2_col]].isna().any(axis=None):
        raise ValueError("POI 层级类别存在缺失")

    provided_pois = set(pois["_poi"])
    events = events[events["_poi"].isin(provided_pois)].copy()
    before_dedup = len(events)
    events = events.drop_duplicates(subset=["_user", "_poi", "_time"]).copy()
    deduplicated_events = before_dedup - len(events)
    events = events.sort_values(["_time", "_user", "_poi"]).reset_index(drop=True)

    train_end = pd.Timestamp(args.train_end)
    if not args.no_validation and not args.validation_end:
        raise ValueError("使用验证集时必须提供 validation_end")
    validation_end = None if args.no_validation else pd.Timestamp(args.validation_end)
    if train_end.tzinfo is None or (validation_end is not None and validation_end.tzinfo is None):
        raise ValueError("时间边界必须显式包含时区")
    train_end = train_end.tz_convert("UTC")
    if validation_end is not None:
        validation_end = validation_end.tz_convert("UTC")
        if validation_end <= train_end:
            raise ValueError("validation_end 必须晚于 train_end")

    train_events = events[events["_time"] <= train_end]
    train_users = set(train_events["_user"])
    train_pois = set(train_events["_poi"])
    catalog_pois = train_pois if args.catalog_scope == "train_seen" else provided_pois
    events = events[events["_user"].isin(train_users) & events["_poi"].isin(catalog_pois)].copy()
    pois = pois[pois["_poi"].isin(catalog_pois)].copy()
    if not len(pois):
        raise ValueError("筛选后目录为空")
    collapsed_consecutive_events = 0
    if args.collapse_consecutive_same_poi:
        events, collapsed_consecutive_events = collapse_consecutive_same_poi(events)

    user_map = stable_mapping(events["_user"])
    poi_map = stable_mapping(pois["_poi"])
    l1_map = stable_mapping(pois[args.category_l1_col])
    l2_map = stable_mapping(pois[args.category_l2_col])
    l2_parent_counts = pois.groupby(args.category_l2_col)[args.category_l1_col].nunique()
    if (l2_parent_counts > 1).any():
        examples = [str(value) for value in l2_parent_counts[l2_parent_counts > 1].index[:10]]
        raise ValueError(f"同一 L2 类别对应多个 L1 类别，请先统一类别树: {examples}")
    pois["pid"] = pois["_poi"].map(poi_map)
    events["UserId"] = events["_user"].map(user_map)
    events["pid"] = events["_poi"].map(poi_map)

    output = args.output_dir
    split_dir = output / "v1_sequence"
    split_dir.mkdir(parents=True, exist_ok=True)
    poi_info = pois[["pid", args.latitude_col, args.longitude_col]].rename(
        columns={args.latitude_col: "latitude", args.longitude_col: "longitude"}
    )
    role_priors = pd.DataFrame(
        {
            "pid": pois["pid"].astype(int),
            "l1_label": pois[args.category_l1_col].astype(str).map(l1_map).astype(int),
            "l2_label": pois[args.category_l2_col].astype(str).map(l2_map).astype(int),
        }
    )
    poi_info.sort_values("pid").to_csv(output / "poi_info.csv", index=False)
    role_priors.sort_values("pid").to_csv(output / "role_priors.csv", index=False)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    max_history = args.max_sequence_length - 1
    for _, group in events.sort_values(["_user", "_time"]).groupby("_user", sort=False):
        records = group.to_dict(orient="records")
        for index in range(args.min_history_length, len(records)):
            target = records[index]
            history = records[max(0, index - max_history) : index]
            split = assign_split(target["_time"], train_end, validation_end)
            sequence = history + [target]
            split_rows[split].append(
                {
                    "UserId": int(target["UserId"]),
                    "target_utc": target["_time"].isoformat(),
                    "sequence_PoiId": str([int(item["pid"]) for item in sequence]),
                    "sequence_UTCTimeOffset": str(
                        [local_time_text(item["_time"], int(item["_offset"])) for item in sequence]
                    ),
                }
            )

    split_names = {
        "train": "train_poi_sequence.csv",
        "validation": "validation_poi_sequence.csv",
        "test": "test_poi_sequence.csv",
    }
    enabled_splits = ("train", "test") if args.no_validation else tuple(split_names)
    for split in enabled_splits:
        filename = split_names[split]
        frame = pd.DataFrame(split_rows[split])
        if frame.empty:
            raise ValueError(f"{split} 切分没有有效样本，请检查时间边界")
        frame = frame.sort_values(["target_utc", "UserId"]).reset_index(drop=True)
        frame.to_csv(split_dir / filename, index=False)

    mappings = {
        "user_id_to_internal": user_map,
        "poi_id_to_internal": poi_map,
        "category_l1_to_internal": l1_map,
        "category_l2_to_internal": l2_map,
    }
    (output / "id_mappings.json").write_text(
        json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "events_source": str(args.events),
        "pois_source": str(args.pois),
        "city": args.city or None,
        "event_type": args.event_type or None,
        "train_end_utc": train_end.isoformat(),
        "validation_end_utc": validation_end.isoformat() if validation_end is not None else None,
        "no_validation": args.no_validation,
        "catalog_scope": args.catalog_scope,
        "max_sequence_length": args.max_sequence_length,
        "min_history_length": args.min_history_length,
        "deduplicated_events": deduplicated_events,
        "collapse_consecutive_same_poi": args.collapse_consecutive_same_poi,
        "collapsed_consecutive_events": collapsed_consecutive_events,
        "events_after_catalog_and_state_filter": len(events),
        "users": len(user_map),
        "catalog_pois": len(poi_map),
        "categories_l1": len(l1_map),
        "categories_l2": len(l2_map),
        "samples": {split: len(rows) for split, rows in split_rows.items()},
        "assumptions": {
            "coordinates": "Input coordinates have already been converted to WGS84.",
            "catalog_availability": (
                "Only train-seen POIs are retained."
                if args.catalog_scope == "train_seen"
                else "All supplied POIs are assumed available no later than the training cutoff."
            ),
        },
    }
    (output / "protocol_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
