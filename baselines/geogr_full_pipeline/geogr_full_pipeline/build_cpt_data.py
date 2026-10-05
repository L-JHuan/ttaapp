"""用公开数据可用字段构造 GeoGR 多模板 CPT 语料。"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from geogr_full_pipeline.build_em_data import load_codes
from geogr_full_pipeline.common import (
    load_public_catalog,
    public_poi_description,
    write_json,
)
from geogr_full_pipeline.sid_utils import sid_tokens


def format_trajectory(
    poi_sequence: list[int],
    time_sequence: list[str],
    pid_to_sid: dict[int, str],
) -> str:
    events = [
        f"{time} visited {pid_to_sid[pid]}"
        for pid, time in zip(poi_sequence, time_sequence)
    ]
    return (
        "User behavior trajectory modeling. Chronological POI visits: "
        + ", ".join(events)
        + "."
    )


def build_rows(
    poi_info: Path,
    role_priors: Path,
    id_mappings: Path | None,
    train_sequences: Path,
    sid_csv: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    catalog = load_public_catalog(poi_info, role_priors, id_mappings)
    codes = load_codes(sid_csv)
    if {poi.pid for poi in catalog} != set(codes):
        raise ValueError("目录 POI 与 CPT SID 覆盖范围不一致")
    pid_to_sid = {pid: sid_tokens(code) for pid, code in codes.items()}

    rows: list[dict[str, str]] = []
    for poi in catalog:
        sid = pid_to_sid[poi.pid]
        description = public_poi_description(poi)
        rows.extend(
            [
                {
                    "template": "poi_description_generation",
                    "text": (
                        f"POI description generation. POI SID: {sid}. "
                        f"POI description: {description}"
                    ),
                },
                {
                    "template": "poi_structured_information_alignment",
                    "text": (
                        "POI structured information alignment. "
                        f"{description} Corresponding POI SID: {sid}."
                    ),
                },
                {
                    "template": "poi_qa_definition",
                    "text": (
                        f"POI QA definition. Question: What public attributes are "
                        f"associated with POI SID {sid}? Answer: {description}"
                    ),
                },
            ]
        )

    frame = pd.read_csv(train_sequences)
    required = {"sequence_PoiId", "sequence_UTCTimeOffset"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"{train_sequences} 缺少字段: {missing}")
    skipped = Counter()
    for row in frame.itertuples(index=False):
        try:
            poi_sequence = [int(value) for value in ast.literal_eval(str(row.sequence_PoiId))]
            time_sequence = [str(value) for value in ast.literal_eval(str(row.sequence_UTCTimeOffset))]
        except (SyntaxError, ValueError, TypeError):
            skipped["parse"] += 1
            continue
        if not poi_sequence or len(poi_sequence) != len(time_sequence):
            skipped["shape"] += 1
            continue
        if any(pid not in pid_to_sid for pid in poi_sequence):
            skipped["unknown_poi"] += 1
            continue
        rows.append(
            {
                "template": "user_behavior_trajectory_modeling",
                "text": format_trajectory(poi_sequence, time_sequence, pid_to_sid),
            }
        )

    counts = Counter(row["template"] for row in rows)
    report = {
        "catalog_pois": len(catalog),
        "train_sequence_rows": int(len(frame)),
        "samples": len(rows),
        "skipped_parse": int(skipped["parse"]),
        "skipped_shape": int(skipped["shape"]),
        "skipped_unknown_poi": int(skipped["unknown_poi"]),
        **{f"template_{name}": int(count) for name, count in sorted(counts.items())},
    }
    return rows, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build matched-protocol GeoGR CPT data")
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--train_sequences", type=Path, required=True)
    parser.add_argument("--sid_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = build_rows(
        args.poi_info,
        args.role_priors,
        args.id_mappings,
        args.train_sequences,
        args.sid_csv,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_json(
        args.report_json,
        {
            "status": "GEOGR_FULL_CPT_DATA_OK",
            "scope": "training split only; public fields only",
            "sid_tokenization": "original Llama tokenizer; no added atomic SID tokens",
            "output_json": str(args.output_json),
            **report,
        },
    )


if __name__ == "__main__":
    main()
