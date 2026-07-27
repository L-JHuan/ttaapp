from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SID_PATTERN = re.compile(r"<[a-z]_\d+>")
SID_LABELS = "abcdefghijklmnopqrstuvwxyz"


class SidTrie:
    def __init__(self, sid_token_ids: list[list[int]], eos_token_id: int):
        self.root: dict[int, dict] = {}
        self.eos_token_id = int(eos_token_id)
        for ids in sid_token_ids:
            node = self.root
            for token_id in ids:
                node = node.setdefault(int(token_id), {})
            node[self.eos_token_id] = {}

    def allowed(self, generated_ids: list[int]) -> list[int]:
        node = self.root
        for token_id in generated_ids:
            token_id = int(token_id)
            if token_id == self.eos_token_id:
                return [self.eos_token_id]
            if token_id not in node:
                return [self.eos_token_id]
            node = node[token_id]
        allowed_ids = list(node.keys())
        return allowed_ids if allowed_ids else [self.eos_token_id]


def sid_atoms(text: Any) -> list[str]:
    if isinstance(text, list):
        text = "".join(str(item) for item in text)
    return SID_PATTERN.findall(str(text or ""))


def canonical_sid(text: Any) -> str:
    return "".join(sid_atoms(text))


def sid_values_to_text(values: list[int]) -> str:
    if len(values) > len(SID_LABELS):
        raise ValueError(f"SID has too many levels: {values}")
    return "".join(f"<{SID_LABELS[index]}_{value}>" for index, value in enumerate(values))


def sid_text_from_row(row: pd.Series) -> str:
    if "sid_tokens" in row and pd.notna(row["sid_tokens"]):
        sid = canonical_sid(str(row["sid_tokens"]))
        if sid:
            return sid
    if "sid" in row and pd.notna(row["sid"]):
        raw = str(row["sid"]).strip()
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed:
            return sid_values_to_text([int(x) for x in parsed])
        sid = canonical_sid(raw)
        if sid:
            return sid
    return ""


def load_sid_trie(path: Path, tokenizer: Any) -> tuple[SidTrie, set[str], dict[str, list[int]]]:
    df = pd.read_csv(path)
    token_paths: list[list[int]] = []
    legal_sids: set[str] = set()
    seen_tokens: set[tuple[int, ...]] = set()
    for _, row in df.iterrows():
        sid = sid_text_from_row(row)
        if not sid:
            continue
        legal_sids.add(sid)
        key = tuple(int(x) for x in tokenizer.encode(sid, add_special_tokens=False))
        if key and key not in seen_tokens:
            seen_tokens.add(key)
            token_paths.append(list(key))
    if not token_paths:
        raise ValueError(f"No valid SID in {path}")
    return SidTrie(token_paths, tokenizer.eos_token_id), legal_sids, {}


def format_prompt(row: dict[str, Any]) -> str:
    return (
        f"### Instruction:\n{str(row['instruction']).strip()}\n\n"
        f"### Input:\n{str(row['input']).strip()}\n\n"
        "### Response:\n"
    )


def metrics(rows: list[dict[str, Any]], predictions: list[list[str]]) -> dict[str, Any]:
    hits = {1: 0, 5: 0, 10: 0}
    ndcg10 = 0.0
    unique_sum = 0
    for row, preds in zip(rows, predictions):
        gold = canonical_sid(row.get("output", row.get("gold", "")))
        seen: set[str] = set()
        unique: list[str] = []
        for pred in preds:
            sid = canonical_sid(pred)
            if sid and sid not in seen:
                seen.add(sid)
                unique.append(sid)
            if len(unique) >= 10:
                break
        unique_sum += len(unique)
        for k in hits:
            if gold in unique[:k]:
                hits[k] += 1
        if gold in unique[:10]:
            rank = unique.index(gold) + 1
            ndcg10 += 1.0 / math.log2(rank + 1)
    denom = max(len(rows), 1)
    return {
        "recall@1": hits[1] / denom,
        "recall@5": hits[5] / denom,
        "recall@10": hits[10] / denom,
        "ndcg@10": ndcg10 / denom,
        "avg_unique_predictions": unique_sum / denom,
        "samples": len(rows),
    }


def load_model(base_model: str, adapter_dir: str, tokenizer_path: str, device: str) -> tuple[Any, Any]:
    tok_path = tokenizer_path or adapter_dir or base_model
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer
