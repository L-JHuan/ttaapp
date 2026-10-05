"""使用 Llama-3 的 beam 候选执行 GeoGR EM-style 三层 SID 更新。"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from geogr_full_pipeline.build_em_data import EM_INSTRUCTION
from geogr_full_pipeline.common import load_public_catalog, public_poi_description, write_json
from geogr_full_pipeline.sid_utils import sid_tokens


SID_PATTERN = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")


@dataclass(frozen=True)
class Candidate:
    code: tuple[int, int, int]
    score: float
    rank: int


def format_prompt(description: str) -> str:
    return (
        f"### Instruction:\n{EM_INSTRUCTION}\n\n"
        f"### Input:\n{description}\n\n"
        "### Response:\n"
    )


def parse_code(text: str, codebook_size: int) -> tuple[int, int, int] | None:
    match = SID_PATTERN.search(str(text))
    if match is None:
        return None
    code = tuple(int(match.group(index)) for index in range(1, 4))
    if any(value < 0 or value >= codebook_size for value in code):
        return None
    return code


def load_codebook(path: Path) -> dict[int, tuple[int, int, int]]:
    frame = pd.read_csv(path)
    if missing := sorted({"pid", "sid"} - set(frame.columns)):
        raise ValueError(f"{path} 缺少字段: {missing}")
    result: dict[int, tuple[int, int, int]] = {}
    for row in frame.itertuples(index=False):
        values = tuple(int(value) for value in ast.literal_eval(str(row.sid)))
        if len(values) != 3:
            raise ValueError(f"pid={row.pid} 不是三层 SID")
        result[int(row.pid)] = values
    return result


def stable_candidate_assignment(
    candidates: dict[int, list[Candidate]],
) -> tuple[dict[int, tuple[int, int, int]], list[int]]:
    """候选 SID 保留最高分申请者，其余 POI 继续申请下一候选。"""
    preferences = {
        pid: sorted(rows, key=lambda item: (-item.score, item.rank, item.code))
        for pid, rows in candidates.items()
    }
    next_index = {pid: 0 for pid in preferences}
    queue = deque(sorted(preferences))
    held: dict[tuple[int, int, int], tuple[int, Candidate]] = {}
    exhausted: set[int] = set()
    while queue:
        pid = queue.popleft()
        index = next_index[pid]
        if index >= len(preferences[pid]):
            exhausted.add(pid)
            continue
        candidate = preferences[pid][index]
        next_index[pid] += 1
        current = held.get(candidate.code)
        if current is None:
            held[candidate.code] = (pid, candidate)
            continue
        current_pid, current_candidate = current
        challenger_key = (candidate.score, -candidate.rank, -pid)
        current_key = (
            current_candidate.score,
            -current_candidate.rank,
            -current_pid,
        )
        if challenger_key > current_key:
            held[candidate.code] = (pid, candidate)
            queue.append(current_pid)
        else:
            queue.append(pid)
    assigned = {pid: code for code, (pid, _candidate) in held.items()}
    missing = sorted(set(candidates) - set(assigned) | exhausted)
    return assigned, missing


def local_fallbacks(
    prior: tuple[int, int, int], codebook_size: int
) -> Iterable[tuple[int, int, int]]:
    yield prior
    for position in (2, 1, 0):
        for value in range(codebook_size):
            if value == prior[position]:
                continue
            code = list(prior)
            code[position] = value
            yield tuple(code)
    for first, second in ((1, 2), (0, 2), (0, 1)):
        for first_value in range(codebook_size):
            for second_value in range(codebook_size):
                code = list(prior)
                code[first] = first_value
                code[second] = second_value
                yield tuple(code)


def complete_unique_assignment(
    candidates: dict[int, list[Candidate]],
    prior_codes: dict[int, tuple[int, int, int]],
    codebook_size: int,
) -> tuple[dict[int, tuple[int, int, int]], dict[str, int]]:
    if set(candidates) != set(prior_codes):
        raise ValueError("候选与初始 SID 的 POI 集合不一致")
    assigned, missing = stable_candidate_assignment(candidates)
    used = set(assigned.values())
    local_fallback_count = 0
    global_fallback_count = 0
    global_codes = (
        (a, b, c)
        for a in range(codebook_size)
        for b in range(codebook_size)
        for c in range(codebook_size)
    )
    for pid in missing:
        selected = next(
            (code for code in local_fallbacks(prior_codes[pid], codebook_size) if code not in used),
            None,
        )
        if selected is not None:
            local_fallback_count += 1
        else:
            selected = next((code for code in global_codes if code not in used), None)
            if selected is None:
                raise ValueError("三层 SID 空间不足以完成唯一分配")
            global_fallback_count += 1
        assigned[pid] = selected
        used.add(selected)
    if len(assigned) != len(prior_codes) or len(set(assigned.values())) != len(assigned):
        raise AssertionError("EM SID 最终分配不唯一")
    candidate_codes = {
        pid: {candidate.code for candidate in rows} for pid, rows in candidates.items()
    }
    return assigned, {
        "beam_assigned": sum(assigned[pid] in candidate_codes[pid] for pid in assigned),
        "local_fallback_assigned": local_fallback_count,
        "global_fallback_assigned": global_fallback_count,
        "retained_prior": sum(assigned[pid] == prior_codes[pid] for pid in assigned),
        "changed": sum(assigned[pid] != prior_codes[pid] for pid in assigned),
    }


class CombinationTrie:
    """在原始 Llama tokenizer 上约束所有合法的三层码组合。"""

    def __init__(self, tokenizer, codebook_size: int) -> None:
        self.root: dict[int, dict] = {}
        self.eos_token_id = int(tokenizer.eos_token_id)
        layer_ids = {
            prefix: {
                value: tokenizer.encode(
                    f"<{prefix}_{value}>", add_special_tokens=False
                )
                for value in range(codebook_size)
            }
            for prefix in ("a", "b", "c")
        }
        composable = True
        for left, right in (("a", "b"), ("b", "c")):
            for left_value in range(codebook_size):
                for right_value in range(codebook_size):
                    joint = tokenizer.encode(
                        f"<{left}_{left_value}><{right}_{right_value}>",
                        add_special_tokens=False,
                    )
                    separate = (
                        layer_ids[left][left_value] + layer_ids[right][right_value]
                    )
                    if joint != separate:
                        composable = False
                        break
                if not composable:
                    break
            if not composable:
                break
        for first in range(codebook_size):
            for second in range(codebook_size):
                for third in range(codebook_size):
                    node = self.root
                    if composable:
                        token_ids = (
                            layer_ids["a"][first]
                            + layer_ids["b"][second]
                            + layer_ids["c"][third]
                        )
                    else:
                        token_ids = tokenizer.encode(
                            sid_tokens([first, second, third]),
                            add_special_tokens=False,
                        )
                    for token_id in token_ids:
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
        return list(node) if node else [self.eos_token_id]


@torch.no_grad()
def generate_candidates(
    model,
    tokenizer,
    sid_trie: CombinationTrie,
    description: str,
    codebook_size: int,
    num_beams: int,
    cutoff_len: int,
    no_cache: bool,
) -> list[Candidate]:
    prompt = format_prompt(description)
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
    def prefix_allowed_tokens_fn(_batch_id: int, ids: torch.Tensor) -> list[int]:
        generated = ids[prompt_len:].tolist()
        return sid_trie.allowed(generated)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=20,
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
    scores = output.sequences_scores.detach().float().cpu().tolist()
    rows: list[Candidate] = []
    seen: set[tuple[int, int, int]] = set()
    for rank, (sequence, score) in enumerate(zip(output.sequences, scores), start=1):
        response = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=False)
        code = parse_code(response, codebook_size)
        if code is not None and code not in seen:
            seen.add(code)
            rows.append(Candidate(code=code, score=float(score), rank=rank))
    return rows


def load_candidate_json(path: Path) -> dict[int, list[Candidate]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(pid): [
            Candidate(tuple(int(value) for value in row["code"]), float(row["score"]), int(row["rank"]))
            for row in rows
        ]
        for pid, rows in payload["candidates"].items()
    }


def write_codebook(path: Path, codes: dict[int, tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pid", "sid", "sid_tokens"])
        writer.writeheader()
        for pid in sorted(codes):
            values = list(codes[pid])
            writer.writerow({"pid": pid, "sid": str(values), "sid_tokens": sid_tokens(values)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GeoGR EM-style SID refinement")
    parser.add_argument("--poi_info", type=Path, required=True)
    parser.add_argument("--role_priors", type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--input_sid_csv", type=Path, required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_sid_csv", type=Path, required=True)
    parser.add_argument("--candidates_json", type=Path, required=True)
    parser.add_argument("--report_json", type=Path, required=True)
    parser.add_argument("--reuse_candidates", action="store_true")
    parser.add_argument("--candidates_only", action="store_true")
    parser.add_argument("--codebook_size", type=int, required=True)
    parser.add_argument("--num_beams", type=int, default=20)
    parser.add_argument("--cutoff_len", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index/num_shards 非法")
    if args.num_shards > 1 and not args.candidates_only:
        raise ValueError("分片候选生成必须设置 --candidates_only")
    prior_codes = load_codebook(args.input_sid_csv)
    catalog = load_public_catalog(args.poi_info, args.role_priors, args.id_mappings)
    if {poi.pid for poi in catalog} != set(prior_codes):
        raise ValueError("目录与输入 SID 覆盖范围不一致")
    if args.reuse_candidates:
        candidates = load_candidate_json(args.candidates_json)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": args.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if len(tokenizer) != model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.adapter_dir)
        model.eval()
        sid_trie = CombinationTrie(tokenizer, args.codebook_size)
        selected_catalog = catalog[args.shard_index :: args.num_shards]
        if args.limit > 0:
            selected_catalog = selected_catalog[: args.limit]
        candidates = {}
        for poi in tqdm(selected_catalog, desc="EM beam candidates", mininterval=60):
            rows = generate_candidates(
                model,
                tokenizer,
                sid_trie,
                public_poi_description(poi),
                args.codebook_size,
                args.num_beams,
                args.cutoff_len,
                args.no_cache,
            )
            candidates[poi.pid] = rows
        write_json(
            args.candidates_json,
            {
                "status": "GEOGR_FULL_EM_CANDIDATES_OK",
                "num_beams": args.num_beams,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "candidates": {
                    str(pid): [
                        {"code": list(row.code), "score": row.score, "rank": row.rank}
                        for row in rows
                    ]
                    for pid, rows in candidates.items()
                },
            },
        )
    if args.candidates_only:
        gpu_memory = {}
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            gpu_memory = {
                "allocated_gib": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gib": torch.cuda.memory_reserved() / (1024**3),
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
            }
        write_json(
            args.report_json,
            {
                "status": "GEOGR_FULL_EM_CANDIDATE_SMOKE_OK",
                "samples": len(candidates),
                "num_beams": args.num_beams,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "candidate_size_histogram": dict(
                    sorted(Counter(len(rows) for rows in candidates.values()).items())
                ),
                "gpu_memory": gpu_memory,
                "candidates_json": str(args.candidates_json),
            },
        )
        return
    if set(candidates) != set(prior_codes):
        raise ValueError("正式 EM 更新必须覆盖完整目录；limit 仅可用于候选生成 smoke")
    final_codes, assignment_report = complete_unique_assignment(
        candidates,
        prior_codes,
        args.codebook_size,
    )
    write_codebook(args.output_sid_csv, final_codes)
    candidate_size_hist = Counter(len(rows) for rows in candidates.values())
    write_json(
        args.report_json,
        {
            "status": "GEOGR_FULL_EM_REFINEMENT_OK",
            "catalog_pois": len(final_codes),
            "codebook_size": args.codebook_size,
            "num_beams": args.num_beams,
            "candidate_size_histogram": dict(sorted(candidate_size_hist.items())),
            "assignment": assignment_report,
            "final_unique_paths": len(set(final_codes.values())),
            "final_sid_length": 3,
            "collision_leaf_added": False,
            "input_sid_csv": str(args.input_sid_csv),
            "output_sid_csv": str(args.output_sid_csv),
        },
    )


if __name__ == "__main__":
    main()
