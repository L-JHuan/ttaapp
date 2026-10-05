"""在统一 Llama-3/LoRA 协议下执行 GeoGR 多模板继续预训练。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


class CptDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, cutoff_len: int) -> None:
        self.items: list[tuple[int, ...]] = []
        self.skipped_empty = 0
        for row in rows:
            text = str(row.get("text", "")).strip()
            if not text:
                self.skipped_empty += 1
                continue
            token_ids = tokenizer(
                text + tokenizer.eos_token,
                add_special_tokens=True,
                truncation=True,
                max_length=cutoff_len,
            )["input_ids"]
            if len(token_ids) >= 2:
                self.items.append(tuple(int(value) for value in token_ids))
        if not self.items:
            raise ValueError("CPT 数据中没有可训练文本")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[int, ...]:
        return self.items[index]


def make_collate(tokenizer):
    def collate(items: list[tuple[int, ...]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item) for item in items)
        input_ids = torch.full(
            (len(items), max_len), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        for row_index, item in enumerate(items):
            length = len(item)
            values = torch.tensor(item, dtype=torch.long)
            input_ids[row_index, :length] = values
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = values
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} 必须是 JSON 列表")
    rows = [row for row in payload if isinstance(row, dict)]
    return rows[:limit] if limit > 0 else rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoGR matched-protocol CPT trainer")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--cutoff_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "grad_accum",
        "num_train_epochs",
        "cutoff_len",
        "logging_steps",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} 必须为正数")
    if args.learning_rate <= 0 or args.warmup_steps < 0:
        raise ValueError("learning_rate 必须为正数，warmup_steps 不能为负数")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} 已存在且非空")


def setup_distributed() -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), rank, world_size, True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, False


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device, rank, world_size, distributed = setup_distributed()
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        rows = load_rows(args.dataset, args.limit)
        dataset = CptDataset(rows, tokenizer, args.cutoff_len)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=make_collate(tokenizer),
            pin_memory=device.type == "cuda",
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.lora_target_modules.split(","),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.to(device)
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[device.index], output_device=device.index
            )

        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        updates_per_epoch = math.ceil(len(loader) / args.grad_accum)
        total_updates = updates_per_epoch * args.num_train_epochs
        if args.max_steps > 0:
            total_updates = min(total_updates, args.max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(args.warmup_steps, total_updates),
            num_training_steps=max(total_updates, 1),
        )

        global_step = 0
        epoch_losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, args.num_train_epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            losses: list[float] = []
            progress = tqdm(
                loader,
                desc=f"CPT epoch {epoch}",
                disable=rank != 0,
                mininterval=60,
            )
            for micro_step, batch in enumerate(progress, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch, use_cache=False, return_dict=True)
                loss = outputs.loss / args.grad_accum
                loss.backward()
                losses.append(float(outputs.loss.detach().float().cpu()))
                should_update = micro_step % args.grad_accum == 0 or micro_step == len(loader)
                if not should_update:
                    continue
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % args.logging_steps == 0:
                    print(
                        f"epoch={epoch} step={global_step} loss={sum(losses[-args.logging_steps:]) / min(len(losses), args.logging_steps):.6f}",
                        flush=True,
                    )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break
            epoch_losses.append(sum(losses) / max(len(losses), 1))
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        if rank == 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            adapter_dir = args.output_dir / "final_cpt"
            unwrapped = model.module if distributed else model
            unwrapped.save_pretrained(adapter_dir, safe_serialization=False)
            tokenizer.save_pretrained(adapter_dir)
            (args.output_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "status": "GEOGR_FULL_CPT_OK",
                        "scope": "matched Llama-3-8B LoRA protocol",
                        "dataset": str(args.dataset),
                        "samples": len(dataset),
                        "skipped_empty": dataset.skipped_empty,
                        "world_size": world_size,
                        "effective_batch_size": args.batch_size
                        * args.grad_accum
                        * world_size,
                        "global_step": global_step,
                        "epoch_losses": epoch_losses,
                        "args": vars(args),
                        "adapter_dir": str(adapter_dir),
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        if distributed:
            torch.distributed.barrier()
    finally:
        if distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
