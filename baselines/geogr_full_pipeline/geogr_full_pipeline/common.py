"""GeoGR 全流程适配共享的数据与 SID 工具。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from geogr_full_pipeline.sid_utils import CatalogPoi, load_catalog


GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(latitude: float, longitude: float, precision: int = 7) -> str:
    """不依赖外部库，将 WGS84 坐标确定性编码为 geohash。"""
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError(f"非法坐标: ({latitude}, {longitude})")
    if precision <= 0:
        raise ValueError("precision 必须为正数")
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits = (16, 8, 4, 2, 1)
    output: list[str] = []
    value = 0
    bit_index = 0
    use_longitude = True
    while len(output) < precision:
        bounds = lon_range if use_longitude else lat_range
        coordinate = longitude if use_longitude else latitude
        midpoint = (bounds[0] + bounds[1]) / 2.0
        if coordinate >= midpoint:
            value |= bits[bit_index]
            bounds[0] = midpoint
        else:
            bounds[1] = midpoint
        use_longitude = not use_longitude
        if bit_index < 4:
            bit_index += 1
        else:
            output.append(GEOHASH_ALPHABET[value])
            bit_index = 0
            value = 0
    return "".join(output)


def public_poi_description(poi: CatalogPoi, geohash_precision: int = 7) -> str:
    """仅使用公开 NYC/TKY 真实可用字段构造 POI 描述。"""
    geohash = encode_geohash(poi.latitude, poi.longitude, geohash_precision)
    return (
        f"POI coordinates: latitude {poi.latitude:.6f}, longitude {poi.longitude:.6f}; "
        f"geohash: {geohash}; category level 1: {poi.category_l1}; "
        f"category level 2: {poi.category_l2}."
    )


def load_public_catalog(
    poi_info: Path,
    role_priors: Path,
    id_mappings: Path | None,
) -> list[CatalogPoi]:
    return load_catalog(poi_info, role_priors, id_mappings)


def save_embeddings(path: Path, pids: list[int], embeddings: np.ndarray) -> None:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) != len(pids):
        raise ValueError("pid 数量与嵌入矩阵不一致")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pids=np.asarray(pids, dtype=np.int64),
        embeddings=values,
    )


def load_embeddings(path: Path) -> tuple[list[int], np.ndarray]:
    payload = np.load(path)
    if "pids" not in payload or "embeddings" not in payload:
        raise ValueError(f"{path} 缺少 pids 或 embeddings")
    pids = [int(value) for value in payload["pids"].tolist()]
    embeddings = np.asarray(payload["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or len(pids) != len(embeddings):
        raise ValueError("嵌入文件形状非法")
    if len(set(pids)) != len(pids) or not np.isfinite(embeddings).all():
        raise ValueError("嵌入文件包含重复 pid 或非有限数值")
    return pids, embeddings


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
