"""TAP-SID 监督微调入口。

The verified training objective is:

    L = L_rec + alpha_prefix * reliability_weight * L_z1

TAP-SID 使用五级路径 ``<a_coarse><b_fine><c_l1><d_l2><e_leaf>``。
正式复现配置令 ``alpha_prefix=0``；兼容头仅用于保持已验证训练实现的
初始化顺序和 checkpoint 格式，不参与正式损失。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from tap_sid.json_records import load_json_records


SID_PATTERN = re.compile(r"<[a-z]_\d+>")
EVENT_PATTERN = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}(?::\d{2})?) visited (?P<sid>(?:<[a-z]_\d+>)+)"
)


@dataclass(frozen=True)
class ReliabilityConfig:
    enabled: bool
    recent_window: int
    entropy_strong: float
    entropy_weak: float
    tail_support_le: int
    head_support_ge: int
    strong_weight: float
    normal_weight: float
    weak_weight: float
    normalize: bool


@dataclass(frozen=True)
class EncodedItem:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_position: int
    z1_label: int
    z2_label: int
    z12_label: int
    sibling_z2_mask: int
    reliability_group: str
    reliability_weight: float


def sid_atoms(text: Any) -> list[str]:
    if isinstance(text, list):
        text = "".join(str(item) for item in text)
    return SID_PATTERN.findall(str(text or ""))


def canonical_sid(text: Any) -> str:
    return "".join(sid_atoms(text))


def format_prompt(row: dict[str, Any]) -> str:
    return (
        f"### Instruction:\n{str(row['instruction']).strip()}\n\n"
        f"### Input:\n{str(row['input']).strip()}\n\n"
        "### Response:\n"
    )


def load_rows(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = load_json_records(path, limit=limit)
    for index, row in enumerate(rows):
        missing = [key for key in ("instruction", "input", "output") if key not in row]
        if missing:
            raise ValueError(f"{path} row {index} missing fields: {missing}")
    return rows


class DistributedEvalSampler(Sampler[int]):
    """在多进程验证时无填充地划分样本，避免重复样本影响验证损失。"""

    def __init__(self, dataset: Dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


def build_z1_vocab(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        atoms = sid_atoms(row.get("output", ""))
        if atoms:
            counts[atoms[0]] += 1
    labels = sorted(counts, key=lambda label: (-counts[label], label))
    if not labels:
        raise ValueError("No z1 labels found in training rows")
    return {label: index for index, label in enumerate(labels)}, dict(counts)


def build_z2_vocab(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        atoms = sid_atoms(row.get("output", ""))
        if len(atoms) > 1:
            counts[atoms[1]] += 1
    labels = sorted(counts, key=lambda label: (-counts[label], label))
    if not labels:
        raise ValueError("No z2 labels found in training rows")
    return {label: index for index, label in enumerate(labels)}, dict(counts)


def build_z12_vocab(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        atoms = sid_atoms(row.get("output", ""))
        if len(atoms) > 1:
            counts[atoms[0] + atoms[1]] += 1
    labels = sorted(counts, key=lambda label: (-counts[label], label))
    if not labels:
        raise ValueError("No z12 labels found in training rows")
    return {label: index for index, label in enumerate(labels)}, dict(counts)


def build_parent_child_ids(
    rows: list[dict[str, Any]],
    z1_label_to_id: dict[str, int],
    z2_label_to_id: dict[str, int],
) -> tuple[list[list[int]], dict[str, dict[str, int]]]:
    parent_child_counts: dict[str, Counter[str]] = {
        label: Counter() for label in z1_label_to_id
    }
    for row in rows:
        atoms = sid_atoms(row.get("output", ""))
        if len(atoms) > 1 and atoms[0] in z1_label_to_id and atoms[1] in z2_label_to_id:
            parent_child_counts[atoms[0]][atoms[1]] += 1

    parent_child_ids: list[list[int]] = [[] for _ in range(len(z1_label_to_id))]
    parent_child_label_counts: dict[str, dict[str, int]] = {}
    for parent_label, parent_id in z1_label_to_id.items():
        child_counts = parent_child_counts[parent_label]
        ordered_children = sorted(child_counts, key=lambda label: (-child_counts[label], label))
        parent_child_ids[parent_id] = [int(z2_label_to_id[label]) for label in ordered_children]
        parent_child_label_counts[parent_label] = {
            label: int(child_counts[label]) for label in ordered_children
        }
    if any(not children for children in parent_child_ids):
        empty = [label for label, index in z1_label_to_id.items() if not parent_child_ids[index]]
        raise ValueError(f"Some z1 parents have no z2 children: {empty[:10]}")
    return parent_child_ids, parent_child_label_counts


def make_legal_child_mask(parent_child_ids: list[list[int]], num_z2_labels: int) -> torch.Tensor:
    mask = torch.zeros((len(parent_child_ids), num_z2_labels), dtype=torch.bool)
    for parent_id, child_ids in enumerate(parent_child_ids):
        if child_ids:
            mask[parent_id, torch.tensor(child_ids, dtype=torch.long)] = True
    return mask


def history_z1s(input_text: str) -> list[str]:
    z1s: list[str] = []
    for match in EVENT_PATTERN.finditer(str(input_text or "")):
        atoms = sid_atoms(match.group("sid"))
        if atoms:
            z1s.append(atoms[0])
    return z1s


def normalized_entropy(labels: list[str]) -> float:
    if not labels:
        return 1.0
    counts = Counter(labels)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        entropy -= p * math.log(max(p, 1e-12))
    return float(entropy / max(math.log(len(counts)), 1e-12)) if len(counts) > 1 else 0.0


def mode_label(labels: list[str]) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    return sorted(counts, key=lambda label: (-counts[label], label))[0]


def support_bucket_by_count(
    label: str,
    counts: dict[str, int],
    tail_support_le: int,
    head_support_ge: int,
) -> str:
    support = int(counts.get(label, 0))
    if tail_support_le > 0 and support <= tail_support_le:
        return "tail"
    if head_support_ge > 0 and support >= head_support_ge:
        return "head"
    return "mid"


def compute_reliability(
    gold_z1: str,
    input_text: str,
    z1_counts: dict[str, int],
    config: ReliabilityConfig,
) -> tuple[str, float]:
    if not config.enabled:
        return "normal", 1.0
    z1s = history_z1s(input_text)
    recent = z1s[-config.recent_window :] if config.recent_window > 0 else z1s
    recent_count = sum(1 for label in recent if label == gold_z1)
    full_count = sum(1 for label in z1s if label == gold_z1)
    last_z1 = z1s[-1] if z1s else ""
    history_mode = mode_label(z1s)
    entropy = normalized_entropy(z1s)
    support_bucket = support_bucket_by_count(
        gold_z1,
        z1_counts,
        config.tail_support_le,
        config.head_support_ge,
    )
    strong = (
        last_z1 == gold_z1
        and (
            recent_count >= 2
            or entropy <= config.entropy_strong
            or history_mode == gold_z1
        )
    ) or (recent_count >= 2 and history_mode == gold_z1)
    weak = (recent_count == 0 and entropy >= config.entropy_weak) or (
        support_bucket == "tail" and full_count == 0
    )
    if strong:
        return "strong", float(config.strong_weight)
    if weak:
        return "weak", float(config.weak_weight)
    return "normal", float(config.normal_weight)


class PrefixAuxDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        z1_label_to_id: dict[str, int],
        z2_label_to_id: dict[str, int],
        z12_label_to_id: dict[str, int],
        z1_counts: dict[str, int],
        reliability: ReliabilityConfig,
        cutoff_len: int,
        sibling_min_support: int = 5,
    ) -> None:
        self.items: list[EncodedItem] = []
        self.skipped: Counter[str] = Counter()
        self.z12_label_to_id = z12_label_to_id
        for row in rows:
            item = self.encode_row(
                row,
                tokenizer,
                z1_label_to_id,
                z2_label_to_id,
                z12_label_to_id,
                z1_counts,
                reliability,
                cutoff_len,
                sibling_min_support,
            )
            if item is not None:
                self.items.append(item)
        if not self.items:
            raise ValueError("No trainable rows after encoding")

    def encode_row(
        self,
        row: dict[str, Any],
        tokenizer: Any,
        z1_label_to_id: dict[str, int],
        z2_label_to_id: dict[str, int],
        z12_label_to_id: dict[str, int],
        z1_counts: dict[str, int],
        reliability: ReliabilityConfig,
        cutoff_len: int,
        sibling_min_support: int,
    ) -> EncodedItem | None:
        output_sid = canonical_sid(row.get("output", ""))
        atoms = sid_atoms(output_sid)
        if len(atoms) < 2:
            self.skipped["bad_sid"] += 1
            return None
        z1 = atoms[0]
        z2 = atoms[1]
        z12 = atoms[0] + atoms[1]
        if z1 not in z1_label_to_id or z2 not in z2_label_to_id:
            self.skipped["unknown_prefix"] += 1
            return None
        z12_label = z12_label_to_id.get(z12, -1)

        prompt = format_prompt(row)
        response = f"{output_sid}<|eot_id|>"
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        prompt_budget = cutoff_len - len(response_ids)
        if prompt_budget <= 1:
            self.skipped["too_long_response"] += 1
            return None
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=prompt_budget,
        )["input_ids"]
        if not prompt_ids:
            self.skipped["empty_prompt"] += 1
            return None

        input_ids = list(prompt_ids) + list(response_ids)
        labels = [-100] * len(prompt_ids) + list(response_ids)
        prompt_position = len(prompt_ids) - 1
        group, weight = compute_reliability(z1, str(row.get("input", "")), z1_counts, reliability)
        sibling_mask = int(z1_counts.get(z1, 0) >= int(sibling_min_support))
        return EncodedItem(
            input_ids=tuple(int(x) for x in input_ids),
            labels=tuple(int(x) for x in labels),
            prompt_position=int(prompt_position),
            z1_label=int(z1_label_to_id[z1]),
            z2_label=int(z2_label_to_id[z2]),
            z12_label=int(z12_label),
            sibling_z2_mask=sibling_mask,
            reliability_group=group,
            reliability_weight=float(weight),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> EncodedItem:
        return self.items[index]


def reliability_summary(items: list[EncodedItem], normalizer: float) -> dict[str, Any]:
    groups: dict[str, Counter[str]] = {"strong": Counter(), "normal": Counter(), "weak": Counter()}
    total = Counter()
    denom = normalizer if normalizer > 0 else 1.0
    for item in items:
        group = item.reliability_group if item.reliability_group in groups else "normal"
        raw = float(item.reliability_weight)
        groups[group]["samples"] += 1
        groups[group]["raw_weight_sum"] += raw
        groups[group]["weight_sum"] += raw / denom
        total["samples"] += 1
        total["raw_weight_sum"] += raw
        total["weight_sum"] += raw / denom
    samples = max(float(total["samples"]), 1.0)
    return {
        "samples": int(total["samples"]),
        "raw_weight_mean": float(total["raw_weight_sum"]) / samples,
        "weight_mean": float(total["weight_sum"]) / samples,
        "groups": {
            group: {
                "samples": int(counter["samples"]),
                "ratio": float(counter["samples"]) / samples,
                "raw_weight_mean": float(counter["raw_weight_sum"]) / max(float(counter["samples"]), 1.0),
                "weight_mean": float(counter["weight_sum"]) / max(float(counter["samples"]), 1.0),
            }
            for group, counter in groups.items()
        },
    }


def make_collate(tokenizer: Any, reliability_normalizer: float):
    def collate(items: list[EncodedItem]) -> dict[str, torch.Tensor | list[str]]:
        max_len = max(len(item.input_ids) for item in items)
        input_ids = torch.full((len(items), max_len), tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((len(items), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        prompt_positions = torch.zeros(len(items), dtype=torch.long)
        z1_labels = torch.zeros(len(items), dtype=torch.long)
        z2_labels = torch.zeros(len(items), dtype=torch.long)
        z12_labels = torch.full((len(items),), -1, dtype=torch.long)
        sibling_z2_mask = torch.zeros(len(items), dtype=torch.float)
        reliability_weights = torch.ones(len(items), dtype=torch.float)
        groups: list[str] = []
        denom = reliability_normalizer if reliability_normalizer > 0 else 1.0
        for row, item in enumerate(items):
            length = len(item.input_ids)
            input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
            labels[row, :length] = torch.tensor(item.labels, dtype=torch.long)
            attention_mask[row, :length] = 1
            prompt_positions[row] = int(item.prompt_position)
            z1_labels[row] = int(item.z1_label)
            z2_labels[row] = int(item.z2_label)
            z12_labels[row] = int(item.z12_label)
            sibling_z2_mask[row] = float(item.sibling_z2_mask)
            reliability_weights[row] = float(item.reliability_weight) / denom
            groups.append(item.reliability_group)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_positions": prompt_positions,
            "z1_labels": z1_labels,
            "z2_labels": z2_labels,
            "z12_labels": z12_labels,
            "sibling_z2_mask": sibling_z2_mask,
            "reliability_weights": reliability_weights,
            "reliability_groups": groups,
        }

    return collate


class TapSidModel(nn.Module):
    def __init__(
        self,
        lm: nn.Module,
        hidden_size: int,
        num_z1_labels: int,
        num_z2_labels: int,
        num_z12_labels: int,
        legal_child_mask: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lm = lm
        self.prefix_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_z1_labels),
        )
        self.z2_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_z2_labels),
        )
        self.z12_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_z12_labels),
        )
        self.sibling_parent_embedding = nn.Embedding(num_z1_labels, hidden_size)
        self.sibling_z2_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_z2_labels),
        )
        self.register_buffer("legal_child_mask", legal_child_mask.bool(), persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_positions: torch.Tensor,
        z1_labels: torch.Tensor,
        z2_labels: torch.Tensor,
        z12_labels: torch.Tensor,
        sibling_z2_mask: torch.Tensor,
        reliability_weights: torch.Tensor,
        lm_loss_weight: float,
        alpha_prefix: float,
    ) -> dict[str, torch.Tensor]:
        try:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        except TypeError:
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        prompt_hidden = outputs.hidden_states[-1][batch_index, prompt_positions]
        z1_logits = self.prefix_head(prompt_hidden.float())
        z2_logits = self.z2_head(prompt_hidden.float())
        z12_logits = self.z12_head(prompt_hidden.float())
        sibling_hidden = prompt_hidden.float() + self.sibling_parent_embedding(z1_labels).float()
        raw_sibling_z2_logits = self.sibling_z2_head(sibling_hidden)
        legal_mask = self.legal_child_mask[z1_labels]
        sibling_z2_logits = raw_sibling_z2_logits.masked_fill(
            ~legal_mask,
            torch.finfo(raw_sibling_z2_logits.dtype).min,
        )
        prefix_per_sample = F.cross_entropy(z1_logits, z1_labels, reduction="none")
        reliability_weights = reliability_weights.to(prefix_per_sample.device, prefix_per_sample.dtype)
        prefix_loss = (prefix_per_sample * reliability_weights).mean()
        z2_loss = F.cross_entropy(z2_logits, z2_labels)
        z12_valid = z12_labels >= 0
        if z12_valid.any():
            z12_loss = F.cross_entropy(z12_logits[z12_valid], z12_labels[z12_valid])
        else:
            z12_loss = z12_logits.sum() * 0.0
        gold_child_legal = legal_mask[batch_index, z2_labels]
        sibling_valid = (sibling_z2_mask > 0) & gold_child_legal
        if sibling_valid.any():
            sibling_z2_loss = F.cross_entropy(sibling_z2_logits[sibling_valid], z2_labels[sibling_valid])
        else:
            sibling_z2_loss = raw_sibling_z2_logits.sum() * 0.0
        lm_loss = outputs.loss
        # 旧 v6 的零权重辅助头仍需挂到计算图上，否则 DDP 会判定参数未使用。
        zero_weight_aux = 0.0 * (z2_loss + z12_loss + sibling_z2_loss)
        loss = float(lm_loss_weight) * lm_loss + float(alpha_prefix) * prefix_loss + zero_weight_aux
        return {
            "loss": loss,
            "lm_loss": lm_loss.detach(),
            "prefix_loss": prefix_loss.detach(),
            "prefix_loss_unweighted": prefix_per_sample.mean().detach(),
            "z2_loss": z2_loss.detach(),
            "z12_loss": z12_loss.detach(),
            "sibling_z2_loss": sibling_z2_loss.detach(),
            "z1_accuracy": (z1_logits.argmax(dim=-1) == z1_labels).float().mean().detach(),
            "z2_accuracy": (z2_logits.argmax(dim=-1) == z2_labels).float().mean().detach(),
            "z12_accuracy": (
                (z12_logits[z12_valid].argmax(dim=-1) == z12_labels[z12_valid]).float().mean()
                if z12_valid.any()
                else z12_logits.new_tensor(0.0)
            ).detach(),
        }


def dist_info() -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size, world_size > 1


def setup_device(backend: str) -> tuple[torch.device, int, int, bool]:
    rank, local_rank, world_size, distributed = dist_info()
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=backend)
        return torch.device("cuda", local_rank), rank, world_size, True
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), rank, world_size, False


def cleanup_distributed(distributed: bool) -> None:
    if distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lm_and_tokenizer(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    tok_path = args.tokenizer_path or args.init_adapter_dir or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    lm = AutoModelForCausalLM.from_pretrained(args.base_model, **kwargs)
    if args.init_adapter_dir:
        lm = PeftModel.from_pretrained(lm, args.init_adapter_dir, is_trainable=True)
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        lm = get_peft_model(lm, config)
    if args.gradient_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(lm, "enable_input_require_grads"):
            lm.enable_input_require_grads()
        lm.config.use_cache = False
    lm.to(device)
    return lm, tokenizer


def maybe_load_prefix_head(model: TapSidModel, path_text: str, rank: int) -> None:
    if not path_text:
        return
    path = Path(path_text)
    if path.is_dir():
        path = path / "prefix_head.pt"
    payload = torch.load(path, map_location="cpu")
    state = payload.get("prefix_head_state")
    if not state:
        raise ValueError(f"{path} missing prefix_head_state")
    missing, unexpected = model.prefix_head.load_state_dict(state, strict=False)
    optional_heads = [
        ("z2_head_state", model.z2_head),
        ("z12_head_state", model.z12_head),
        ("sibling_parent_embedding_state", model.sibling_parent_embedding),
        ("sibling_z2_head_state", model.sibling_z2_head),
    ]
    optional_report: dict[str, dict[str, list[str]]] = {}
    for payload_key, module in optional_heads:
        state_dict = payload.get(payload_key)
        if state_dict:
            opt_missing, opt_unexpected = module.load_state_dict(state_dict, strict=False)
            optional_report[payload_key] = {
                "missing": list(opt_missing),
                "unexpected": list(opt_unexpected),
            }
    if rank == 0:
        print(
            f"Loaded prefix head from {path}; missing={missing}, "
            f"unexpected={unexpected}, optional={optional_report}"
        )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lm_loss_weight: float,
    alpha_prefix: float,
    distributed: bool,
    rank: int,
) -> dict[str, float]:
    eval_model = model.module if isinstance(model, DistributedDataParallel) else model
    eval_model.eval()
    metric_keys = ("loss", "lm_loss", "prefix_loss", "prefix_loss_unweighted", "z1_accuracy")
    sums = Counter()
    total = 0
    for batch in tqdm(loader, desc="eval", mininterval=30, disable=rank != 0):
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        out = eval_model(
            **tensor_batch,
            lm_loss_weight=lm_loss_weight,
            alpha_prefix=alpha_prefix,
        )
        batch_size = int(tensor_batch["input_ids"].shape[0])
        total += batch_size
        for key in metric_keys:
            sums[key] += float(out[key].item()) * batch_size
    packed = torch.tensor(
        [float(sums[key]) for key in metric_keys] + [float(total)],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
    eval_model.train()
    denom = max(float(packed[-1].item()), 1.0)
    return {
        key: float(packed[index].item()) / denom
        for index, key in enumerate(metric_keys)
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    rank: int,
    distributed: bool,
    global_step: int,
) -> tuple[int, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(loader, desc=f"train epoch {epoch}", mininterval=30, disable=rank != 0)
    sums = Counter()
    epoch_sums = Counter()
    metric_keys = (
        "loss",
        "lm_loss",
        "prefix_loss",
        "prefix_loss_unweighted",
        "z1_accuracy",
        "z2_loss",
        "z2_accuracy",
        "z12_loss",
        "z12_accuracy",
        "sibling_z2_loss",
    )
    total_samples = 0
    epoch_samples = 0
    since = time.time()
    for step, batch in enumerate(iterator, start=1):
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        should_step = step % args.grad_accum == 0 or step == len(loader)
        sync_context = nullcontext()
        if distributed and not should_step:
            sync_context = model.no_sync()
        with sync_context:
            out = model(
                **tensor_batch,
                lm_loss_weight=args.lm_loss_weight,
                alpha_prefix=args.alpha_prefix,
            )
            loss = out["loss"] / args.grad_accum
            loss.backward()
        batch_size = int(tensor_batch["input_ids"].shape[0])
        total_samples += batch_size
        epoch_samples += batch_size
        for key in metric_keys:
            sums[key] += float(out[key].item()) * batch_size
            epoch_sums[key] += float(out[key].item()) * batch_size
        if should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and global_step % args.logging_steps == 0:
                elapsed = max(time.time() - since, 1e-6)
                avg = {key: sums[key] / max(float(total_samples), 1.0) for key in metric_keys}
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "samples_seen": total_samples,
                            "items_per_sec": total_samples / elapsed,
                            **avg,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                sums.clear()
                total_samples = 0
                since = time.time()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
    summary = {
        key: epoch_sums[key] / max(float(epoch_samples), 1.0)
        for key in metric_keys
    }
    return global_step, summary


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def select_best_epoch(epoch_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not epoch_summaries:
        raise ValueError("epoch_summaries 不能为空")
    validated = []
    for summary in epoch_summaries:
        valid = summary.get("valid")
        if isinstance(valid, dict) and math.isfinite(float(valid.get("lm_loss", math.nan))):
            validated.append(summary)
    if validated:
        selected = min(
            validated,
            key=lambda item: (float(item["valid"]["lm_loss"]), int(item["epoch"])),
        )
        return {
            "epoch": int(selected["epoch"]),
            "metric": "lm_loss",
            "value": float(selected["valid"]["lm_loss"]),
            "mode": "min",
        }
    selected = max(epoch_summaries, key=lambda item: int(item["epoch"]))
    return {
        "epoch": int(selected["epoch"]),
        "metric": None,
        "value": None,
        "mode": "final",
    }


def prefix_head_payload(
    model: nn.Module,
    z1_label_to_id: dict[str, int],
    z1_counts: dict[str, int],
    z2_label_to_id: dict[str, int],
    z2_counts: dict[str, int],
    z12_label_to_id: dict[str, int],
    z12_counts: dict[str, int],
    parent_child_ids: list[list[int]],
    parent_child_label_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "prefix_head_state": raw_model.prefix_head.state_dict(),
        "z2_head_state": raw_model.z2_head.state_dict(),
        "z12_head_state": raw_model.z12_head.state_dict(),
        "sibling_parent_embedding_state": raw_model.sibling_parent_embedding.state_dict(),
        "sibling_z2_head_state": raw_model.sibling_z2_head.state_dict(),
        "legal_child_mask": raw_model.legal_child_mask.detach().cpu(),
        "label_to_id": z1_label_to_id,
        "label_counts": z1_counts,
        "z1_label_to_id": z1_label_to_id,
        "z1_label_counts": z1_counts,
        "z2_label_to_id": z2_label_to_id,
        "z2_label_counts": z2_counts,
        "z12_label_to_id": z12_label_to_id,
        "z12_label_counts": z12_counts,
        "parent_child_ids": parent_child_ids,
        "parent_child_label_counts": parent_child_label_counts,
    }


def save_epoch_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    output_dir: Path,
    epoch: int,
    z1_label_to_id: dict[str, int],
    z1_counts: dict[str, int],
    z2_label_to_id: dict[str, int],
    z2_counts: dict[str, int],
    z12_label_to_id: dict[str, int],
    z12_counts: dict[str, int],
    parent_child_ids: list[list[int]],
    parent_child_label_counts: dict[str, dict[str, int]],
) -> Path:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint_dir = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    raw_model.lm.save_pretrained(checkpoint_dir, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        prefix_head_payload(
            model,
            z1_label_to_id,
            z1_counts,
            z2_label_to_id,
            z2_counts,
            z12_label_to_id,
            z12_counts,
            parent_child_ids,
            parent_child_label_counts,
        ),
        checkpoint_dir / "prefix_head.pt",
    )
    return checkpoint_dir


def finalize_selected_checkpoint(
    output_dir: Path,
    selected_checkpoint: Path,
    train_summary: dict[str, Any],
) -> None:
    final_dir = output_dir / "final_sft"
    if final_dir.exists():
        raise FileExistsError(final_dir)
    shutil.copytree(selected_checkpoint, final_dir)
    shutil.copy2(final_dir / "prefix_head.pt", output_dir / "prefix_head.pt")
    (output_dir / "training_summary.json").write_text(
        json.dumps(json_safe(train_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TAP-SID coarse-to-fine region SID prefix auxiliary SFT trainer.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--init_adapter_dir", default="")
    parser.add_argument("--init_prefix_head_path", default="")
    parser.add_argument("--train_dataset", type=Path, required=True)
    parser.add_argument("--valid_dataset", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    parser.add_argument("--alpha_prefix", type=float, default=0.05)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--z1_reliability_weighted", action="store_true")
    parser.add_argument("--z1_reliability_recent_window", type=int, default=5)
    parser.add_argument("--z1_reliability_entropy_strong", type=float, default=0.55)
    parser.add_argument("--z1_reliability_entropy_weak", type=float, default=0.85)
    parser.add_argument("--z1_reliability_tail_support_le", type=int, default=0)
    parser.add_argument("--z1_reliability_head_support_ge", type=int, default=0)
    parser.add_argument("--z1_reliability_strong_weight", type=float, default=1.3)
    parser.add_argument("--z1_reliability_normal_weight", type=float, default=1.0)
    parser.add_argument("--z1_reliability_weak_weight", type=float, default=0.7)
    parser.add_argument("--z1_reliability_normalize", action="store_true")
    parser.add_argument("--sibling_min_support", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    parser.add_argument("--eval_during_train", action="store_true")
    parser.add_argument("--attn_implementation", default="")
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_dataset_paths(
    train_dataset: Path,
    valid_dataset: Path | None,
    eval_during_train: bool,
) -> None:
    if not train_dataset.exists():
        raise FileNotFoundError(train_dataset)
    if valid_dataset is not None and not valid_dataset.exists():
        raise FileNotFoundError(valid_dataset)
    if eval_during_train and valid_dataset is None:
        raise ValueError("--eval_during_train requires --valid_dataset")


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = ("batch_size", "eval_batch_size", "grad_accum", "num_train_epochs", "cutoff_len", "logging_steps")
    for name in positive_ints:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning_rate must be > 0")
    for name in (
        "weight_decay",
        "lm_loss_weight",
        "alpha_prefix",
        "head_dropout",
        "lora_dropout",
        "z1_reliability_entropy_strong",
        "z1_reliability_entropy_weak",
        "z1_reliability_strong_weight",
        "z1_reliability_normal_weight",
        "z1_reliability_weak_weight",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if args.alpha_prefix < 0:
        raise ValueError("--alpha_prefix must be >= 0 for w/o z1 auxiliary ablation")
    if args.z1_reliability_recent_window <= 0:
        raise ValueError("--z1_reliability_recent_window must be > 0")
    if args.sibling_min_support < 0:
        raise ValueError("--sibling_min_support must be >= 0")
    if args.limit_train < 0 or args.limit_val < 0 or args.max_steps < 0:
        raise ValueError("--limit_train/--limit_val/--max_steps must be >= 0")
    validate_dataset_paths(args.train_dataset, args.valid_dataset, args.eval_during_train)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{args.output_dir} already exists; pass --overwrite to reuse")


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device, rank, world_size, distributed = setup_device(args.dist_backend)
    try:
        train_rows = load_rows(args.train_dataset, args.limit_train)
        valid_rows = load_rows(args.valid_dataset, args.limit_val) if args.valid_dataset is not None else []
        z1_label_to_id, z1_counts = build_z1_vocab(train_rows)
        z2_label_to_id, z2_counts = build_z2_vocab(train_rows)
        z12_label_to_id, z12_counts = build_z12_vocab(train_rows)
        parent_child_ids, parent_child_label_counts = build_parent_child_ids(
            train_rows,
            z1_label_to_id,
            z2_label_to_id,
        )
        legal_child_mask = make_legal_child_mask(parent_child_ids, len(z2_label_to_id))
        reliability = ReliabilityConfig(
            enabled=bool(args.z1_reliability_weighted),
            recent_window=args.z1_reliability_recent_window,
            entropy_strong=args.z1_reliability_entropy_strong,
            entropy_weak=args.z1_reliability_entropy_weak,
            tail_support_le=args.z1_reliability_tail_support_le,
            head_support_ge=args.z1_reliability_head_support_ge,
            strong_weight=args.z1_reliability_strong_weight,
            normal_weight=args.z1_reliability_normal_weight,
            weak_weight=args.z1_reliability_weak_weight,
            normalize=bool(args.z1_reliability_normalize),
        )
        lm, tokenizer = load_lm_and_tokenizer(args, device)
        hidden_size = int(getattr(lm.config, "hidden_size", 4096))
        model = TapSidModel(
            lm=lm,
            hidden_size=hidden_size,
            num_z1_labels=len(z1_label_to_id),
            num_z2_labels=len(z2_label_to_id),
            num_z12_labels=len(z12_label_to_id),
            legal_child_mask=legal_child_mask,
            dropout=args.head_dropout,
        ).to(device)
        maybe_load_prefix_head(model, args.init_prefix_head_path, rank)

        train_dataset = PrefixAuxDataset(
            train_rows,
            tokenizer,
            z1_label_to_id,
            z2_label_to_id,
            z12_label_to_id,
            z1_counts,
            reliability,
            args.cutoff_len,
            args.sibling_min_support,
        )
        valid_dataset = (
            PrefixAuxDataset(
                valid_rows,
                tokenizer,
                z1_label_to_id,
                z2_label_to_id,
                z12_label_to_id,
                z1_counts,
                reliability,
                args.cutoff_len,
                args.sibling_min_support,
            )
            if valid_rows
            else None
        )
        raw_weight_mean = reliability_summary(train_dataset.items, 1.0)["raw_weight_mean"]
        normalizer = raw_weight_mean if args.z1_reliability_normalize and raw_weight_mean > 0 else 1.0
        collate = make_collate(tokenizer, normalizer)
        train_sampler = (
            DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            if distributed
            else None
        )
        valid_sampler = (
            DistributedEvalSampler(valid_dataset, rank=rank, world_size=world_size)
            if distributed and valid_dataset is not None
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        valid_loader = (
            DataLoader(
                valid_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                sampler=valid_sampler,
                collate_fn=collate,
                pin_memory=device.type == "cuda",
            )
            if valid_dataset is not None
            else None
        )
        if distributed:
            model = DistributedDataParallel(model, device_ids=[device.index], output_device=device.index)
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "train_samples": len(train_dataset),
                        "valid_samples": len(valid_dataset) if valid_dataset is not None else 0,
                        "world_size": world_size,
                        "z1_labels": len(z1_label_to_id),
                        "z2_labels": len(z2_label_to_id),
                        "z12_labels": len(z12_label_to_id),
                        "parent_child_nonempty": sum(1 for children in parent_child_ids if children),
                        "reliability": reliability_summary(train_dataset.items, normalizer),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        global_step = 0
        epoch_summaries: list[dict[str, Any]] = []
        for epoch in range(1, args.num_train_epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            global_step, epoch_summary = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args,
                epoch,
                rank,
                distributed,
                global_step,
            )
            valid_metrics = None
            if args.eval_during_train:
                if valid_loader is None:
                    raise RuntimeError("Validation loader is unavailable")
                valid_metrics = evaluate(
                    model,
                    valid_loader,
                    device,
                    args.lm_loss_weight,
                    args.alpha_prefix,
                    distributed,
                    rank,
                )
            if rank == 0:
                epoch_record = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train": epoch_summary,
                }
                if valid_metrics is not None:
                    epoch_record["valid"] = valid_metrics
                    print(
                        f"eval epoch {epoch}: {json.dumps(valid_metrics, ensure_ascii=False)}",
                        flush=True,
                    )
                checkpoint_dir = save_epoch_checkpoint(
                    model,
                    tokenizer,
                    args.output_dir,
                    epoch,
                    z1_label_to_id,
                    z1_counts,
                    z2_label_to_id,
                    z2_counts,
                    z12_label_to_id,
                    z12_counts,
                    parent_child_ids,
                    parent_child_label_counts,
                )
                epoch_record["checkpoint"] = str(checkpoint_dir)
                epoch_summaries.append(epoch_record)
                print(f"Saved epoch {epoch} checkpoint -> {checkpoint_dir}", flush=True)
            if distributed:
                torch.distributed.barrier()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        if rank == 0:
            checkpoint_selection = select_best_epoch(epoch_summaries)
            selected_epoch = int(checkpoint_selection["epoch"])
            selected_summary = next(
                item for item in epoch_summaries if int(item["epoch"]) == selected_epoch
            )
            selected_checkpoint = Path(selected_summary["checkpoint"])
            summary = {
                "model_type": "tap_sid_sft",
                "args": vars(args),
                "train_config": {
                    "base_model": args.base_model,
                    "tokenizer_path": args.tokenizer_path,
                    "train_dataset": str(args.train_dataset),
                    "valid_dataset": str(args.valid_dataset) if args.valid_dataset is not None else None,
                    "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum,
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "world_size": world_size,
                    "seed": args.seed,
                },
                "global_step": global_step,
                "z1_labels": len(z1_label_to_id),
                "z2_labels": len(z2_label_to_id),
                "z12_labels": len(z12_label_to_id),
                "label_vocab": {
                    "z1": len(z1_label_to_id),
                    "z2": len(z2_label_to_id),
                    "z12": len(z12_label_to_id),
                },
                "train_samples": len(train_dataset),
                "valid_samples": len(valid_dataset) if valid_dataset is not None else 0,
                "train_skipped": dict(train_dataset.skipped),
                "valid_skipped": dict(valid_dataset.skipped) if valid_dataset is not None else {},
                "reliability": reliability_summary(train_dataset.items, normalizer),
                "epoch_summaries": epoch_summaries,
                "checkpoint_selection": checkpoint_selection,
                "best_epoch": selected_epoch,
                "best_checkpoint": str(selected_checkpoint),
                "final_eval": selected_summary.get("valid"),
            }
            finalize_selected_checkpoint(
                args.output_dir,
                selected_checkpoint,
                summary,
            )
            print(
                f"Selected epoch {selected_epoch}; saved TAP-SID trainer output -> "
                f"{args.output_dir}",
                flush=True,
            )
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
