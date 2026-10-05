"""构造 GeoGR EM 迭代中的 POI 描述到当前 SID 监督数据。"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

from geogr_full_pipeline.common import load_public_catalog, public_poi_description, write_json
from geogr_full_pipeline.sid_utils import sid_tokens


EM_INSTRUCTION = (
    "Given the public attributes of a POI, generate its current three-level "
    "geographic collaborative semantic identifier."
)


def load_codes(path: Path) -> dict[int, list[int]]:
    frame = pd.read_csv(path)
    if missing := sorted({"pid", "sid"} - set(frame.columns)):
        raise ValueError(f"{path} 缺少字段: {missing}")
    result: dict[int, list[int]] = {}
    for row in frame.itertuples(index=False):
        raw = ast.literal_eval(str(row.sid))
        code = [int(value) for value in raw]
        if len(code) != 3:
            raise ValueError(f"pid={row.pid} 不是三层 SID: {code}")
        result[int(row.pid)] = code
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GeoGR EM description-to-SID data")
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--sid_csv", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_public_catalog(args.poi_info, args.role_priors, args.id_mappings)
    codes = load_codes(args.sid_csv)
    catalog_pids = {poi.pid for poi in catalog}
    if catalog_pids != set(codes):
        raise ValueError("目录 POI 与 SID 覆盖范围不一致")
    rows = [
        {
            "instruction": EM_INSTRUCTION,
            "input": public_poi_description(poi),
            "output": sid_tokens(codes[poi.pid]),
            "pid": poi.pid,
        }
        for poi in catalog
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_json(
        args.report_json,
        {
            "status": "GEOGR_FULL_EM_DATA_OK",
            "samples": len(rows),
            "sid_csv": str(args.sid_csv),
            "output_json": str(args.output_json),
        },
    )


if __name__ == "__main__":
    main()
