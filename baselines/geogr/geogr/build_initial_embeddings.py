"""使用外部文本编码器生成 GeoGR 初始 POI 嵌入。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from geogr.build_geogr_sid import encode_descriptions, load_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--encoder_model", required=True)
    parser.add_argument("--encoder_device", default="cuda:0")
    parser.add_argument("--encoder_backend", default="transformers_last_token")
    parser.add_argument("--encoder_dtype", default="auto")
    parser.add_argument("--encoder_batch_size", type=int, default=16)
    parser.add_argument("--encoder_max_length", type=int, default=128)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--report_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.poi_info, args.role_priors, args.id_mappings)
    embeddings = encode_descriptions(
        catalog,
        encoder_model=args.encoder_model,
        device=args.encoder_device,
        backend=args.encoder_backend,
        dtype_name=args.encoder_dtype,
        batch_size=args.encoder_batch_size,
        max_length=args.encoder_max_length,
    )
    pids = np.asarray([poi.pid for poi in catalog], dtype=np.int64)
    if embeddings.shape[0] != len(pids):
        raise ValueError(
            f"嵌入数量与 POI 数不一致: {embeddings.shape[0]} != {len(pids)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("初始嵌入包含 NaN 或 Inf")

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, pids=pids, embeddings=embeddings)
    report = {
        "num_pois": int(len(pids)),
        "embedding_dim": int(embeddings.shape[1]),
        "encoder_model": args.encoder_model,
        "encoder_backend": args.encoder_backend,
        "encoder_dtype": args.encoder_dtype,
        "encoder_max_length": int(args.encoder_max_length),
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
