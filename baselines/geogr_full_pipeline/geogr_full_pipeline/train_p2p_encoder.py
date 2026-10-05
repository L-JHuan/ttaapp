"""使用地理约束共访 POI 对对 Qwen3-Embedding-4B 进行 LoRA 对比微调。"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.nn.functional import all_gather
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from geogr_full_pipeline.sid_utils import (
    GeoPair,
    build_geo_constrained_pairs,
    load_train_user_items,
)
from geogr_full_pipeline.common import (
    load_public_catalog,
    public_poi_description,
    save_embeddings,
    write_json,
)


class PairDataset(Dataset):
    def __init__(self, pairs: list[GeoPair], descriptions: dict[int, str]) -> None:
        self.rows = [
            (descriptions[pair.left], descriptions[pair.right]) for pair in pairs
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.rows[index]


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_indices = (
        attention_mask.size(1)
        - 1
        - attention_mask.flip(dims=(1,)).argmax(dim=1)
    )
    return hidden[
        torch.arange(hidden.size(0), device=hidden.device),
        last_indices,
    ]


def encode_batch(model, tokenizer, texts: list[str], device: torch.device, max_length: int):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    hidden = model(**batch, return_dict=True).last_hidden_state
    return F.normalize(last_token_pool(hidden, batch["attention_mask"]).float(), dim=-1)


def contrastive_loss(
    left: torch.Tensor,
    right: torch.Tensor,
    temperature: float,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if world_size > 1:
        global_left = torch.cat(list(all_gather(left)), dim=0)
        global_right = torch.cat(list(all_gather(right)), dim=0)
        offset = rank * left.shape[0]
    else:
        global_left = left
        global_right = right
        offset = 0
    labels = torch.arange(len(left), device=left.device) + offset
    left_logits = left @ global_right.T / temperature
    right_logits = right @ global_left.T / temperature
    return 0.5 * (
        F.cross_entropy(left_logits, labels) + F.cross_entropy(right_logits, labels)
    )


def setup_distributed(device_text: str) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size, True
    return torch.device(device_text), rank, world_size, False


@torch.no_grad()
def encode_catalog(
    model,
    tokenizer,
    descriptions: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in tqdm(range(0, len(descriptions), batch_size), desc="Encode POIs"):
        vectors = encode_batch(
            model,
            tokenizer,
            descriptions[start : start + batch_size],
            device,
            max_length,
        )
        output.append(vectors.cpu().numpy())
    return np.concatenate(output, axis=0).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoGR P2P LoRA contrastive encoder")
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--train_sequences", type=Path, required=True)
    parser.add_argument("--encoder_model", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--max_distance_km", type=float, default=3.0)
    parser.add_argument("--min_common_users", type=int, default=2)
    parser.add_argument("--swing_alpha", type=float, default=1.0)
    parser.add_argument("--max_pairs_per_poi", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--encode_limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 2
        or args.grad_accum <= 0
        or args.temperature <= 0
        or args.max_steps < 0
        or args.encode_limit < 0
    ):
        raise ValueError(
            "batch_size 必须 >=2，grad_accum/temperature 必须为正数，"
            "max_steps/encode_limit 不能为负数"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device, rank, world_size, distributed = setup_distributed(args.device)

    try:
        catalog = load_public_catalog(args.poi_info, args.role_priors, args.id_mappings)
        descriptions = {poi.pid: public_poi_description(poi) for poi in catalog}
        coordinates = {poi.pid: (poi.latitude, poi.longitude) for poi in catalog}
        user_items = load_train_user_items(args.train_sequences)
        pairs, pair_report = build_geo_constrained_pairs(
            user_items,
            coordinates,
            args.max_distance_km,
            args.min_common_users,
            args.swing_alpha,
            args.max_pairs_per_poi,
        )
        if len(pairs) < 2:
            raise ValueError("地理约束共访正样本对不足")

        tokenizer = AutoTokenizer.from_pretrained(args.encoder_model, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        model = AutoModel.from_pretrained(
            args.encoder_model,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
            ),
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.to(device)
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=args.learning_rate,
        )
        pair_dataset = PairDataset(pairs, descriptions)
        sampler = (
            DistributedSampler(
                pair_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            pair_dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            generator=torch.Generator().manual_seed(args.seed),
            drop_last=True,
        )
        epoch_losses: list[float] = []
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            losses: list[float] = []
            progress = tqdm(
                loader,
                desc=f"P2P epoch {epoch}",
                disable=rank != 0,
                mininterval=60,
            )
            for micro_step, (left_text, right_text) in enumerate(progress, start=1):
                left = encode_batch(model, tokenizer, list(left_text), device, args.max_length)
                right = encode_batch(model, tokenizer, list(right_text), device, args.max_length)
                raw_loss = contrastive_loss(
                    left, right, args.temperature, rank, world_size
                )
                (raw_loss / args.grad_accum).backward()
                losses.append(float(raw_loss.detach().cpu()))
                should_update = micro_step % args.grad_accum == 0 or micro_step == len(loader)
                if not should_update:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            if not losses:
                raise ValueError("对比编码器未产生有效训练 step")
            epoch_losses.append(float(np.mean(losses)))
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = model.module if distributed else model
            adapter_dir = args.output_dir / "encoder_adapter"
            unwrapped.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            encoded_catalog = catalog[: args.encode_limit] if args.encode_limit > 0 else catalog
            ordered_descriptions = [descriptions[poi.pid] for poi in encoded_catalog]
            embeddings = encode_catalog(
                unwrapped,
                tokenizer,
                ordered_descriptions,
                device,
                args.encode_batch_size,
                args.max_length,
            )
            embedding_path = args.output_dir / "refined_embeddings.npz"
            save_embeddings(embedding_path, [poi.pid for poi in encoded_catalog], embeddings)
            gpu_memory = {
                "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
                "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
            write_json(
                args.output_dir / "p2p_encoder_report.json",
                {
                    "status": "GEOGR_FULL_P2P_ENCODER_OK",
                    "scope": "paper-guided matched Llama-3-8B protocol",
                    "catalog_pois": len(catalog),
                    "encoded_catalog_pois": len(encoded_catalog),
                    "description_fields": [
                        "latitude",
                        "longitude",
                        "derived_geohash",
                        "category_l1",
                        "category_l2",
                    ],
                    "unavailable_fields": [
                        "poi_name",
                        "brand",
                        "address",
                        "consumption_level",
                        "active_period",
                    ],
                    "pair_construction": pair_report,
                    "encoder_model": args.encoder_model,
                    "adapter": "LoRA over q/k/v/o projections",
                    "world_size": world_size,
                    "global_contrastive_batch": args.batch_size * world_size,
                    "optimizer_effective_batch": args.batch_size
                    * world_size
                    * args.grad_accum,
                    "epoch_losses": epoch_losses,
                    "global_step": global_step,
                    "gpu_memory_rank0": gpu_memory,
                    "embedding_dimension": int(embeddings.shape[1]),
                    "embedding_path": str(embedding_path),
                },
            )
        if distributed:
            torch.distributed.barrier()
    finally:
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
