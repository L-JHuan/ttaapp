"""将时间切分后的 POI 序列转换为 TAP-SID 监督微调样本。"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


INSTRUCTION = (
    "Here is a record of a user's POI accesses, your task is based on the history "
    "to predict the POI that the user is likely to access at the specified time."
)
LABELS = "abcdefghijklmnopqrstuvwxyz"


def parse_sid(value: Any) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"SID 不是整数列表: {value}")
    return [int(item) for item in value]


def encode_sid(values: list[int]) -> str:
    if len(values) > len(LABELS):
        raise ValueError(f"SID 层数过多: {values}")
    return "".join(f"<{LABELS[index]}_{value}>" for index, value in enumerate(values))


def load_pid_to_sid(path: Path) -> dict[int, str]:
    frame = pd.read_csv(path)
    missing = sorted({"pid", "sid"} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} 缺少字段: {missing}")
    mapping: dict[int, str] = {}
    for _, row in frame.iterrows():
        mapping[int(row["pid"])] = encode_sid(parse_sid(row["sid"]))
    return mapping


def convert_split(
    input_csv: Path,
    output_json: Path,
    pid_to_sid: dict[int, str],
    keep_last_k_per_user: int = 0,
) -> dict[str, int]:
    frame = pd.read_csv(input_csv)
    required = {"UserId", "sequence_PoiId", "sequence_UTCTimeOffset"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{input_csv} 缺少字段: {missing}")
    if "target_utc" in frame.columns:
        frame = frame.sort_values(["target_utc", "UserId"]).reset_index(drop=True)
    if keep_last_k_per_user > 0:
        frame = frame.groupby("UserId", group_keys=False, sort=False).tail(keep_last_k_per_user)
        frame = frame.reset_index(drop=True)

    samples: list[dict[str, str]] = []
    skipped_parse = 0
    skipped_shape = 0
    skipped_unknown = 0
    for row in frame.itertuples(index=False):
        try:
            poi_sequence = ast.literal_eval(str(row.sequence_PoiId))
            time_sequence = ast.literal_eval(str(row.sequence_UTCTimeOffset))
        except (SyntaxError, ValueError):
            skipped_parse += 1
            continue
        if len(poi_sequence) < 2 or len(poi_sequence) != len(time_sequence):
            skipped_shape += 1
            continue

        poi_sequence = [int(pid) for pid in poi_sequence]
        if any(pid not in pid_to_sid for pid in poi_sequence):
            skipped_unknown += 1
            continue
        history = [pid_to_sid[pid] for pid in poi_sequence[:-1]]
        target_sid = pid_to_sid[poi_sequence[-1]]
        history_text = ", ".join(
            f"{time} visited {sid}" for time, sid in zip(time_sequence[:-1], history)
        )
        user_id = row.UserId
        input_text = (
            f"User_{user_id} checkin history: {history_text}.\n"
            f"When {time_sequence[-1]} user_{user_id} is likely to visit:"
        )
        samples.append({"instruction": INSTRUCTION, "input": input_text, "output": target_sid})

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input_rows": int(len(frame)),
        "samples": len(samples),
        "skipped_parse": skipped_parse,
        "skipped_shape": skipped_shape,
        "skipped_unknown_poi": skipped_unknown,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TAP-SID SFT JSON from chronological sequences.")
    parser.add_argument("--sid_csv", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keep_last_k_train", type=int, default=5)
    parser.add_argument("--no_validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pid_to_sid = load_pid_to_sid(args.sid_csv)
    jobs = [
        ("train", "train_poi_sequence.csv", "llm_train.json", args.keep_last_k_train),
        ("test", "test_poi_sequence.csv", "llm_test.json", 0),
    ]
    if not args.no_validation:
        jobs.insert(1, ("validation", "validation_poi_sequence.csv", "llm_val.json", 0))
    report: dict[str, Any] = {"sid_csv": str(args.sid_csv), "splits": {}}
    for split, input_name, output_name, keep_last_k in jobs:
        stats = convert_split(
            args.split_dir / input_name,
            args.output_dir / output_name,
            pid_to_sid,
            keep_last_k_per_user=keep_last_k,
        )
        report["splits"][split] = stats
        print(f"{split}: {stats}")
    (args.output_dir / "llm_json_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
