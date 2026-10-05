"""构建 Spacetime-GR 风格的 geographic block -> inner-POI 两级标识。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class Poi:
    pid: int
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SidRow:
    pid: int
    block_id: int
    inner_id: int
    block_x: int
    block_y: int

    @property
    def sid(self) -> tuple[int, int]:
        return self.block_id, self.inner_id

    @property
    def sid_tokens(self) -> str:
        return f"<a_{self.block_id}><b_{self.inner_id}>"


def _validate_pois(pois: list[Poi]) -> None:
    if not pois:
        raise ValueError("POI 目录为空")
    pids = [poi.pid for poi in pois]
    duplicate_pids = sorted(pid for pid, count in Counter(pids).items() if count > 1)
    if duplicate_pids:
        raise ValueError(f"POI 目录包含重复 pid: {duplicate_pids[:10]}")
    for poi in pois:
        if not math.isfinite(poi.latitude) or not -90.0 <= poi.latitude <= 90.0:
            raise ValueError(f"pid={poi.pid} 的纬度非法: {poi.latitude}")
        if not math.isfinite(poi.longitude) or not -180.0 <= poi.longitude <= 180.0:
            raise ValueError(f"pid={poi.pid} 的经度非法: {poi.longitude}")


def project_to_local_km(
    latitude: float,
    longitude: float,
    reference_latitude: float,
) -> tuple[float, float]:
    """使用固定参考纬度的等距圆柱近似，将经纬度投影到公里坐标。"""
    lat_rad = math.radians(latitude)
    lon_rad = math.radians(longitude)
    ref_rad = math.radians(reference_latitude)
    x = EARTH_RADIUS_KM * lon_rad * math.cos(ref_rad)
    y = EARTH_RADIUS_KM * lat_rad
    return x, y


def assign_hierarchical_codes(
    pois: Iterable[Poi],
    block_size_km: float = 5.0,
    reference_latitude: float | None = None,
) -> tuple[list[SidRow], dict[str, object]]:
    """按固定方格分配 block token，并在每个 block 内按 pid 分配 inner token。"""
    poi_list = list(pois)
    _validate_pois(poi_list)
    if not math.isfinite(block_size_km) or block_size_km <= 0:
        raise ValueError(f"block_size_km 必须为正数: {block_size_km}")
    if reference_latitude is None:
        reference_latitude = float(statistics.median(poi.latitude for poi in poi_list))
    if not -90.0 < reference_latitude < 90.0:
        raise ValueError(f"reference_latitude 非法: {reference_latitude}")

    poi_to_grid: dict[int, tuple[int, int]] = {}
    for poi in poi_list:
        x_km, y_km = project_to_local_km(
            poi.latitude,
            poi.longitude,
            reference_latitude,
        )
        poi_to_grid[poi.pid] = (
            math.floor(x_km / block_size_km),
            math.floor(y_km / block_size_km),
        )

    grids = sorted(set(poi_to_grid.values()), key=lambda value: (value[1], value[0]))
    grid_to_block = {grid: block_id for block_id, grid in enumerate(grids)}
    block_to_pids: dict[int, list[int]] = defaultdict(list)
    for pid, grid in poi_to_grid.items():
        block_to_pids[grid_to_block[grid]].append(pid)

    rows: list[SidRow] = []
    for block_id in sorted(block_to_pids):
        block_x, block_y = grids[block_id]
        for inner_id, pid in enumerate(sorted(block_to_pids[block_id]), start=1):
            rows.append(
                SidRow(
                    pid=pid,
                    block_id=block_id,
                    inner_id=inner_id,
                    block_x=block_x,
                    block_y=block_y,
                )
            )
    rows.sort(key=lambda row: row.pid)

    full_paths = [row.sid for row in rows]
    if len(set(full_paths)) != len(rows):
        raise RuntimeError("block/inner 完整路径不唯一")
    block_sizes = sorted(len(pids) for pids in block_to_pids.values())
    report: dict[str, object] = {
        "status": "SPACETIME_GR_CODEBOOK_OK",
        "implementation": "paper-guided current-protocol adaptation",
        "num_pois": len(rows),
        "num_blocks": len(block_to_pids),
        "block_size_km": float(block_size_km),
        "reference_latitude": float(reference_latitude),
        "projection": "equirectangular with a fixed dataset reference latitude",
        "grid_origin": "global longitude/latitude origin",
        "inner_id_rule": "1..K by ascending pid within each block",
        "full_path_unique": True,
        "block_bucket_size": {
            "min": min(block_sizes),
            "median": float(statistics.median(block_sizes)),
            "mean": float(statistics.fmean(block_sizes)),
            "max": max(block_sizes),
        },
    }
    return rows, report


def _find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    by_lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in fieldnames:
            return candidate
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(f"缺少字段 {candidates}; 当前字段={fieldnames}")


def read_pois(path: Path) -> list[Poi]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} 缺少表头")
        pid_col = _find_column(reader.fieldnames, ("pid", "poi_id", "PoiId"))
        lat_col = _find_column(reader.fieldnames, ("latitude", "lat", "Latitude"))
        lon_col = _find_column(
            reader.fieldnames,
            ("longitude", "lon", "lng", "Longitude"),
        )
        return [
            Poi(
                pid=int(row[pid_col]),
                latitude=float(row[lat_col]),
                longitude=float(row[lon_col]),
            )
            for row in reader
        ]


def write_codebook(path: Path, rows: list[SidRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pid",
                "sid",
                "sid_tokens",
                "a_id",
                "b_id",
                "block_x",
                "block_y",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "pid": row.pid,
                    "sid": str(list(row.sid)),
                    "sid_tokens": row.sid_tokens,
                    "a_id": row.block_id,
                    "b_id": row.inner_id,
                    "block_x": row.block_x,
                    "block_y": row.block_y,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Spacetime-GR geographic block/inner-POI identifier."
    )
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--block_size_km", type=float, default=5.0)
    parser.add_argument("--reference_latitude", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report = assign_hierarchical_codes(
        read_pois(args.poi_info),
        block_size_km=args.block_size_km,
        reference_latitude=args.reference_latitude,
    )
    write_codebook(args.output_csv, rows)
    report.update({"poi_info": str(args.poi_info), "output_csv": str(args.output_csv)})
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
