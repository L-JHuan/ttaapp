"""从训练完成的 RQ-VAE 导出 GNPR Semantic ID。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnpr_baseline.embedding_dataset import EmbeddingDataset
from gnpr_baseline.rqvae.model import CRQVAE


def model_config_from_checkpoint(checkpoint: dict, input_dim: int) -> dict:
    if "config" in checkpoint:
        return dict(checkpoint["config"])
    old_args = checkpoint.get("args")
    if old_args is None:
        raise ValueError("Checkpoint 缺少 config 或旧版 args")
    return {
        "in_dim": input_dim,
        "num_emb_list": list(old_args.num_emb_list),
        "e_dim": int(old_args.e_dim),
        "layers": list(old_args.layers),
        "dropout_prob": float(old_args.dropout_prob),
        "bn": bool(old_args.bn),
        "loss_type": str(old_args.loss_type),
        "quant_loss_weight": float(old_args.quant_loss_weight),
        "beta": float(old_args.beta),
        "kmeans_init": bool(old_args.kmeans_init),
        "kmeans_iters": int(old_args.kmeans_iters),
        "sk_epsilons": list(old_args.sk_epsilons),
        "sk_iters": int(old_args.sk_iters),
        "use_linear": int(old_args.use_linear),
    }


def add_collision_leaf(codes: dict[int, list[int]]) -> dict[int, list[int]]:
    counts = Counter(tuple(code) for code in codes.values())
    next_leaf: Counter[tuple[int, ...]] = Counter()
    result: dict[int, list[int]] = {}
    for pid in sorted(codes):
        code = list(codes[pid])
        key = tuple(code)
        if counts[key] > 1:
            code.append(next_leaf[key])
            next_leaf[key] += 1
        result[pid] = code
    return result


def sid_tokens(code: list[int]) -> str:
    labels = "abcdefghijklmnopqrstuvwxyz"
    return "".join(f"<{labels[index]}_{value}>" for index, value in enumerate(code))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export GNPR residual Semantic IDs.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = EmbeddingDataset(args.data_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = model_config_from_checkpoint(checkpoint, dataset.dim)
    if int(config["in_dim"]) != dataset.dim:
        raise ValueError("Checkpoint 输入维度与 POI embedding 不一致")
    model = CRQVAE(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(args.device)
    model.eval()

    raw_codes: dict[int, list[int]] = {}
    vectors: dict[int, list[float]] = {}
    for pids, inputs in tqdm(loader, desc="export GNPR codebook", mininterval=60):
        quantized, indices = model.get_indices(inputs.to(args.device))
        for row, pid in enumerate(pids.tolist()):
            raw_codes[int(pid)] = [int(value) for value in indices[row].tolist()]
            vectors[int(pid)] = [float(value) for value in quantized[row].tolist()]
    final_codes = add_collision_leaf(raw_codes)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "sid", "sid_tokens", "vector"])
        writer.writeheader()
        for pid in sorted(final_codes):
            writer.writerow(
                {
                    "pid": pid,
                    "sid": str(final_codes[pid]),
                    "sid_tokens": sid_tokens(final_codes[pid]),
                    "vector": json.dumps(vectors[pid]),
                }
            )
    raw_unique = len({tuple(value) for value in raw_codes.values()})
    raw_counts = Counter(tuple(value) for value in raw_codes.values())
    colliding_pois = sum(count for count in raw_counts.values() if count > 1)
    report = {
        "status": "GNPR_CODEBOOK_EXPORT_OK",
        "checkpoint": str(args.checkpoint),
        "pois": len(final_codes),
        "raw_unique_paths": raw_unique,
        "raw_duplicate_excess": len(final_codes) - raw_unique,
        "raw_colliding_pois": colliding_pois,
        "final_unique_paths": len({tuple(value) for value in final_codes.values()}),
        "sid_lengths": dict(Counter(len(value) for value in final_codes.values())),
        "collision_leaf_rule": "within each duplicated residual path, assign 0..n-1 by ascending pid",
    }
    (args.output_csv.parent / "codebook_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
