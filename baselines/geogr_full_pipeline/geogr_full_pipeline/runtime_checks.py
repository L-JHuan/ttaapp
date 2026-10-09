"""检查可复用的训练产物，保留失败阶段文件而不覆盖。"""

import argparse
import json
import math
from pathlib import Path


def finite_summary(value):
    """旧版可能把 NaN 写入成功总结；这样的训练结果不得跳过重训。"""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_summary(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_summary(item) for item in value)
    return True


def checkpoint_ready(checkpoint: Path, final_name: str) -> bool:
    adapter = checkpoint / final_name
    try:
        config = json.loads((adapter / "adapter_config.json").read_text())
        summary = json.loads((checkpoint / "training_summary.json").read_text())
        return isinstance(config, dict) and bool(config) and isinstance(summary, dict) and finite_summary(summary) and int(summary.get("global_step", 0)) > 0 and any(
            path.is_file() and path.stat().st_size > 0
            for path in (adapter / "adapter_model.bin", adapter / "adapter_model.safetensors")
        ) and (adapter / "tokenizer_config.json").is_file()
    except (OSError, ValueError):
        return False


def archive_incomplete_checkpoint(checkpoint: Path, run_root: Path, suffix: str) -> Path:
    root, resolved = run_root.resolve(), checkpoint.resolve()
    if resolved == root or root not in resolved.parents or checkpoint.is_symlink():
        raise ValueError(f"拒绝移动运行目录以外的 checkpoint：{checkpoint}")
    target = checkpoint.with_name(checkpoint.name + ".incomplete_" + suffix)
    if target.exists():
        raise FileExistsError(target)
    checkpoint.rename(target)
    print(f"保留失败训练产物：{target}；只重新执行未完成训练阶段", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--final_name", required=True)
    parser.add_argument("--run_root", type=Path)
    parser.add_argument("--archive_suffix")
    args = parser.parse_args()
    if args.archive_suffix:
        if args.run_root is None:
            parser.error("归档必须指定 run_root")
        if args.checkpoint.exists() and any(args.checkpoint.iterdir()):
            archive_incomplete_checkpoint(args.checkpoint, args.run_root, args.archive_suffix)
    else:
        raise SystemExit(0 if checkpoint_ready(args.checkpoint, args.final_name) else 1)


if __name__ == "__main__":
    main()
