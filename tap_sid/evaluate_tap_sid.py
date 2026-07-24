"""TAP-SID 目录约束生成与 Recall/NDCG 评估入口。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

LLM_DIR = Path(__file__).resolve().parent.parent
if str(LLM_DIR) not in sys.path:
    sys.path.insert(0, str(LLM_DIR))

from tap_sid.catalog_trie import (  # noqa: E402
    canonical_sid,
    format_prompt,
    load_sid_trie,
    metrics,
    sid_atoms,
    load_model,
)
from tap_sid.json_records import load_json_records  # noqa: E402


def read_json(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    return load_json_records(path, limit=limit)


def dedupe_topk(values: list[str], k: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        sid = canonical_sid(value)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
        if len(out) >= k:
            break
    return out


@torch.no_grad()
def generate_one(
    model: Any,
    tokenizer: Any,
    sid_trie: Any,
    prompt: str,
    num_beams: int,
    max_new_tokens: int,
    cutoff_len: int,
    no_cache: bool,
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=cutoff_len,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = int(input_ids.shape[1])

    def prefix_allowed_tokens_fn(batch_id: int, ids: torch.Tensor) -> list[int]:
        generated_ids = ids[prompt_len:].tolist()
        return sid_trie.allowed(generated_ids)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_beams,
        do_sample=False,
        early_stopping=True,
        use_cache=not no_cache,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        return_dict_in_generate=True,
        output_scores=True,
    )
    output_sequence_scores = getattr(output, "sequences_scores", None)
    sequence_scores = (
        output_sequence_scores.detach().float().cpu().tolist()
        if output_sequence_scores is not None
        else [0.0] * int(output.sequences.shape[0])
    )
    rows: list[dict[str, Any]] = []
    for sequence, score in zip(output.sequences, sequence_scores):
        response_ids = sequence[prompt_len:].detach().cpu().tolist()
        response = tokenizer.decode(response_ids, skip_special_tokens=False)
        sid = canonical_sid(response)
        rows.append({"sid": sid, "score": float(score), "raw": response})
    return rows


def run_eval(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    torch.manual_seed(args.seed)
    dataset_rows = read_json(args.dataset, args.limit)
    tokenizer_path = args.tokenizer_path or args.base_model
    model, tokenizer = load_model(args.base_model, args.adapter_dir, tokenizer_path, args.device)
    sid_trie, legal_sids, _ = load_sid_trie(args.semantic_codes, tokenizer)

    predictions: list[list[str]] = []
    output_rows: list[dict[str, Any]] = []
    parse_stats = Counter()
    length_hist = Counter()

    for local_index, row in enumerate(
        tqdm(dataset_rows, desc="TAP-SID constrained eval", mininterval=30)
    ):
        prompt = format_prompt(row)
        generated = generate_one(
            model,
            tokenizer,
            sid_trie,
            prompt,
            args.num_beams,
            args.max_new_tokens,
            args.cutoff_len,
            args.no_cache,
        )
        unique: list[str] = []
        raw_predictions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in generated:
            sid = canonical_sid(item.get("sid", ""))
            if sid and sid not in legal_sids:
                parse_stats["illegal_sid_predictions"] += 1
                continue
            if sid and sid not in seen:
                seen.add(sid)
                unique.append(sid)
                length_hist[str(len(sid_atoms(sid)))] += 1
                raw_predictions.append(
                    {
                        "sid": sid,
                        "score": item["score"],
                        "raw": str(item.get("raw", ""))[:200],
                    }
                )
            if len(unique) >= args.k:
                break
        predictions.append(unique)
        output_row = {
            "gold": row.get("output", row.get("gold", "")),
            "predictions": dedupe_topk(unique, args.k),
            "raw_predictions": raw_predictions,
            "input": row.get("input", ""),
        }
        if "_tap_sample_index" in row:
            output_row["sample_index"] = int(row["_tap_sample_index"])
        elif args.num_shards > 1:
            raise ValueError("多分片评估数据缺少 _tap_sample_index")
        output_rows.append(output_row)

    metric_out = metrics(dataset_rows, predictions)
    metric_out.update(
        {
            "samples": len(dataset_rows),
            "prediction_samples": len(output_rows),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "constrained_sid": True,
            "num_beams": args.num_beams,
            "k": args.k,
            "legal_sid_checked": True,
            "legal_sid_count": len(legal_sids),
            "illegal_sid_predictions": int(parse_stats["illegal_sid_predictions"]),
            "pred_sid_length_histogram": dict(length_hist),
            "config": {
                "base_model": args.base_model,
                "tokenizer_path": tokenizer_path,
                "adapter_dir": args.adapter_dir,
                "dataset": str(args.dataset),
                "semantic_codes": str(args.semantic_codes),
                "cutoff_len": args.cutoff_len,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "device": args.device,
                "limit": args.limit,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
        }
    )
    # 显式保留该字段，便于检查 beam 中有效且不重复的候选数量。
    if "avg_unique_predictions" not in metric_out:
        metric_out["avg_unique_predictions"] = sum(len(x) for x in predictions) / max(len(predictions), 1)
    if not math.isfinite(float(metric_out.get("ndcg@10", 0.0))):
        raise ValueError("Invalid ndcg@10")
    return output_rows, metric_out


def main() -> None:
    parser = argparse.ArgumentParser(description="TAP-SID coarse-to-fine region SID constrained generation eval.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--semantic_codes", type=Path, required=True)
    parser.add_argument("--output_predictions", type=Path, required=True)
    parser.add_argument("--output_metrics", type=Path, required=True)
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--num_beams", type=int, default=10)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index 必须位于 [0, num_shards) 范围内")

    predictions, metric_out = run_eval(args)
    args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_predictions.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_metrics.write_text(json.dumps(metric_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved predictions -> {args.output_predictions}")
    print(f"Saved metrics -> {args.output_metrics}")


if __name__ == "__main__":
    main()
