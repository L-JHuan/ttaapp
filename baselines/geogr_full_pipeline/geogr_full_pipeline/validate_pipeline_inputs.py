"""在 EM 训练之前检查目录、P2P 向量和三层初始 SID 的契约。"""

import argparse
import ast
import json

from pathlib import Path
import pandas as pd
from geogr_full_pipeline.common import load_embeddings
from geogr_full_pipeline.sid_utils import load_catalog


def main():
    parser = argparse.ArgumentParser()
    for name in ("poi_info", "role_priors", "embeddings_npz", "sid_csv", "report_json"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--codebook_size", type=int, required=True)
    args = parser.parse_args()
    catalog = load_catalog(args.poi_info, args.role_priors, args.id_mappings)
    expected = {poi.pid for poi in catalog}
    pids, embeddings = load_embeddings(args.embeddings_npz)
    frame = pd.read_csv(args.sid_csv)
    report = json.loads(args.report_json.read_text())
    if not expected or set(pids) != expected or set(frame["pid"].astype(int)) != expected or frame["pid"].duplicated().any():
        raise ValueError("目录、P2P 向量与初始 SID 的 POI 覆盖不一致")
    if args.codebook_size ** 3 < len(expected):
        raise ValueError("三层码空间不足以在 EM 中分配唯一 POI 标识；请先确认 CODEBOOK_SIZE")
    if report.get("codebook_size") != args.codebook_size or report.get("catalog_pois") != len(expected):
        raise ValueError("初始 RQ 完成报告与当前目录/码本配置不一致")
    for sid in frame["sid"]:
        values = ast.literal_eval(str(sid))
        if len(values) != 3 or any(not isinstance(v, int) or not 0 <= v < args.codebook_size for v in values):
            raise ValueError(f"非法初始 SID：{sid}")
    print(f"INPUT_CONTRACT_OK: catalog={len(expected)} embeddings={embeddings.shape} three-layer SID coverage=100%", flush=True)


if __name__ == "__main__":
    main()
