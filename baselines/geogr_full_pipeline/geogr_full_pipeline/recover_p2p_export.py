"""复用已完成的 P2P adapter，只恢复目录向量导出，不重新训练。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from peft.utils.save_and_load import get_peft_model_state_dict, load_peft_weights
from transformers import AutoModel, AutoTokenizer

from geogr_full_pipeline.common import (
    load_public_catalog, public_poi_description, save_embeddings, write_json,
)
from geogr_full_pipeline.train_p2p_encoder import (
    encode_catalog, input_fingerprints, log_stage,
)


def inspect_saved_adapter(args) -> tuple[dict, dict]:
    """检查权重完整性；缺少完成标记的旧版本仅在入口允许时恢复。"""
    adapter = args.adapter_dir
    for name in ("adapter_config.json", "tokenizer_config.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"保存的 adapter 不完整，缺少 {adapter / name}")
    if not any((adapter / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        raise ValueError("保存的 tokenizer 不完整")
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("task_type") != "FEATURE_EXTRACTION":
        raise ValueError("不是本阶段的 FEATURE_EXTRACTION LoRA adapter")
    if set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise ValueError("adapter 的目标模块与 P2P 训练器不一致")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError("没有可复用的 adapter 权重文件")
    weights = load_peft_weights(str(adapter), device="cpu")
    if not weights or any(not torch.isfinite(value).all() for value in weights.values()):
        raise ValueError("adapter 权重为空或包含非有限数值")
    current_inputs = input_fingerprints(args)
    marker = adapter.parent / "p2p_training_report.json"
    if marker.is_file():
        training = json.loads(marker.read_text(encoding="utf-8"))
        if training.get("training_complete") is not True:
            raise ValueError("训练完成标记无效，禁止复用")
        if training.get("input_sha256") != current_inputs:
            raise ValueError("当前输入与已完成训练的输入不一致")
        if Path(training["encoder_model"]).resolve() != args.encoder_model.resolve():
            raise ValueError("当前基础模型路径与已完成训练不一致")
        if training.get("max_length") != args.max_length or training.get("dtype") != args.dtype:
            raise ValueError("导出长度或精度与训练设置不一致")
        if training.get("encode_limit", 0) != 0:
            raise ValueError("不能将限制目录规模的 smoke adapter 自动视为正式产物")
    else:
        if not args.allow_legacy_adapter:
            raise ValueError(
                "旧 adapter 没有训练完成报告。先确认原日志三轮训练全部完成、"
                "基础模型与输入未变，再显式启用 --allow_legacy_adapter。"
            )
        training = {
            "training_complete": None,
            "completion_evidence": "legacy_adapter_allowed_without_training_completion_marker",
            "input_sha256": None,
            "world_size": None,
            "epoch_losses": None,
            "global_step": None,
            "pair_construction": None,
        }
    return training, weights


def check_loaded_weights(model, saved) -> None:
    """严格核对键、形状和数值，防止 PEFT 警告后保留随机初始化参数。"""
    loaded = get_peft_model_state_dict(model)
    if set(loaded) != set(saved):
        raise ValueError("恢复模型的 adapter 参数键与保存权重不一致")
    for key, value in loaded.items():
        expected = saved[key]
        if value.shape != expected.shape or not torch.equal(
            value.detach().cpu(), expected.to(dtype=value.dtype),
        ):
            raise ValueError(f"恢复的 adapter 权重不一致：{key}")


def parse_args():
    parser = argparse.ArgumentParser(description="GeoGR 已完成 P2P adapter 的恢复导出")
    for name in ("poi_info", "role_priors", "train_sequences", "encoder_model", "adapter_dir", "output_dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--id_mappings", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--encode_batch_size", type=int, default=8)
    parser.add_argument("--allow_legacy_adapter", action="store_true")
    parser.add_argument("--check_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.encode_batch_size <= 0 or args.max_length <= 0:
        raise ValueError("导出 batch 和长度必须为正数")
    log_stage("检查已保存的 P2P adapter 和输入；不会构建共访对或启动训练。")
    training, weights = inspect_saved_adapter(args)
    catalog = load_public_catalog(args.poi_info, args.role_priors, args.id_mappings)
    if training.get("catalog_pois", len(catalog)) != len(catalog):
        raise ValueError("目录规模与训练记录不一致")
    if args.check_only:
        print(json.dumps({
            "status": "P2P_ADAPTER_FILES_CHECK_OK",
            "catalog_pois": len(catalog), "weight_tensors": len(weights),
            "legacy_confirmation": not (args.adapter_dir.parent / "p2p_training_report.json").is_file(),
            "note": "尚未加载基础模型，完整参数形状及数值匹配在实际导出时检查",
        }, ensure_ascii=False), flush=True)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("恢复导出只使用一个进程，不要通过多进程 torchrun 启动")
    for name in ("refined_embeddings.npz", "p2p_encoder_report.json"):
        if (args.output_dir / name).exists():
            raise ValueError(f"禁止覆盖已有导出产物：{args.output_dir / name}")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter_dir), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    base = AutoModel.from_pretrained(
        str(args.encoder_model), local_files_only=True, trust_remote_code=True,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter_dir), is_trainable=False)
    check_loaded_weights(model, weights)
    del weights
    device = torch.device(args.device)
    model.to(device)
    log_stage(f"adapter 权重匹配通过；开始导出 {len(catalog)} 个 POI（无训练通信组）。")
    embeddings = encode_catalog(
        model, tokenizer, [public_poi_description(poi) for poi in catalog],
        device, args.encode_batch_size, args.max_length,
    )
    if embeddings.shape[0] != len(catalog) or not np.isfinite(embeddings).all():
        raise ValueError("导出向量覆盖不完整或包含非有限数值")
    destination = args.output_dir / "refined_embeddings.npz"
    save_embeddings(destination, [poi.pid for poi in catalog], embeddings)
    write_json(args.output_dir / "p2p_encoder_report.json", {
        **training,
        "status": "GEOGR_FULL_P2P_ENCODER_OK",
        "scope": "paper-guided matched Llama-3-8B protocol",
        "recovered_export": True,
        "training_rerun": False,
        "source_adapter": str(args.adapter_dir.resolve()),
        "encoder_model": str(args.encoder_model.resolve()),
        "export_input_sha256": input_fingerprints(args),
        "catalog_pois": len(catalog), "encoded_catalog_pois": len(catalog),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_path": str(destination),
        "adapter_weights_checked": True,
        "export_world_size": 1,
    })
    log_stage(f"P2P 恢复导出成功：{destination}")


if __name__ == "__main__":
    main()
