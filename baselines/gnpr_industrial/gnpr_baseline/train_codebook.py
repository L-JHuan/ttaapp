"""训练 GNPR 的三层 residual-quantization codebook。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_constant_schedule_with_warmup

from gnpr_baseline.embedding_dataset import EmbeddingDataset
from gnpr_baseline.rqvae.model import CRQVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNPR residual codebook.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_step", type=int, default=10)
    parser.add_argument("--min_select_epoch", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--num_emb_list", type=int, nargs="+", default=[64, 64, 64])
    parser.add_argument("--e_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--quant_loss_weight", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--kmeans_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    return parser.parse_args()


@torch.no_grad()
def collision_rate(model: CRQVAE, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    seen: set[tuple[int, ...]] = set()
    sample_count = 0
    for _, vectors in loader:
        _, indices = model.get_indices(vectors.to(device))
        seen.update(tuple(int(value) for value in row) for row in indices.tolist())
        sample_count += len(indices)
    return (sample_count - len(seen)) / sample_count


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EmbeddingDataset(args.data_path)
    if len(dataset) < max(args.num_emb_list):
        raise ValueError("POI 数量少于单层码本大小，无法进行稳定 KMeans 初始化")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    config = {
        "in_dim": dataset.dim,
        "num_emb_list": args.num_emb_list,
        "e_dim": args.e_dim,
        "layers": args.layers,
        "dropout_prob": args.dropout_prob,
        "bn": True,
        "loss_type": "mse",
        "quant_loss_weight": args.quant_loss_weight,
        "beta": args.beta,
        "kmeans_init": True,
        "kmeans_iters": args.kmeans_iters,
        "sk_epsilons": [0.1] * len(args.num_emb_list),
        "sk_iters": 50,
        "use_linear": 1,
    }
    device = torch.device(args.device)
    model = CRQVAE(**config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_constant_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * len(loader),
    )

    best_loss = float("inf")
    best_collision = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = np.zeros(3, dtype=np.float64)
        progress = tqdm(loader, desc=f"codebook epoch {epoch}", mininterval=60)
        for _, vectors in progress:
            vectors = vectors.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, quant_loss, _ = model(vectors, use_sk=False)
            loss, rq_loss, reconstruction_loss = model.compute_loss(
                quant_loss, reconstructed, vectors
            )
            if not torch.isfinite(loss):
                raise ValueError("RQ-VAE loss 出现 NaN 或 Inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            totals += [loss.item(), rq_loss.item(), reconstruction_loss.item()]

        if epoch % args.eval_step != 0:
            continue
        rate = collision_rate(model, eval_loader, device)
        row = {
            "epoch": epoch,
            "loss_sum": float(totals[0]),
            "rq_loss_sum": float(totals[1]),
            "reconstruction_loss_sum": float(totals[2]),
            "collision_rate": float(rate),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        state = {
            "epoch": epoch,
            "config": config,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": row,
            "seed": args.seed,
        }
        torch.save(state, args.output_dir / "latest_model.pth")
        if epoch >= args.min_select_epoch and totals[0] < best_loss:
            best_loss = float(totals[0])
            torch.save(state, args.output_dir / "best_loss_model.pth")
        if epoch >= args.min_select_epoch and rate < best_collision:
            best_collision = float(rate)
            torch.save(state, args.output_dir / "best_collision_model.pth")

    if not (args.output_dir / "best_loss_model.pth").exists():
        raise RuntimeError("未生成 best_loss_model.pth，请检查 epochs 和 min_select_epoch")
    report = {
        "status": "GNPR_CODEBOOK_TRAINING_OK",
        "data_path": str(args.data_path),
        "pois": len(dataset),
        "config": config,
        "epochs": args.epochs,
        "best_loss_sum": best_loss,
        "best_collision_rate": best_collision,
        "selection_checkpoint": "best_loss_model.pth",
        "history": history,
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
