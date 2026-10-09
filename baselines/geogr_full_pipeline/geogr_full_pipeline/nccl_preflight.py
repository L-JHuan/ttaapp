"""在加载大模型前，实际检查各 rank 的 DDP 初始化、梯度和集合通信。"""

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--timeout_seconds", type=int, default=90)
    args = parser.parse_args()
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", local) if args.backend == "nccl" else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if torch.cuda.device_count() != world:
            raise ValueError(f"可见 GPU 数量与 rank 数量不一致：{torch.cuda.device_count()} != {world}")
    print(f"rank={rank}/{world} device={device} preflight starting", flush=True)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout_seconds))
    try:
        model = torch.nn.Linear(256, 256).to(device)
        ddp = DistributedDataParallel(model, device_ids=[local] if device.type == "cuda" else None)
        ddp(torch.ones((2, 256), device=device)).sum().backward()
        # 大于微小控制消息的负载，测试实际训练将使用的通信链路。
        value = torch.full((1024 * 1024,), rank + 1.0, device=device)
        dist.all_reduce(value)
        expected = world * (world + 1) / 2
        if not torch.all(value == expected).item():
            raise RuntimeError("all_reduce 结果不正确")
        dist.broadcast(value, src=0)
        args.report_dir.mkdir(parents=True, exist_ok=True)
        report = {"status": "GEOGR_COMMUNICATION_OK", "rank": rank, "world_size": world,
                  "backend": args.backend, "torch": torch.__version__,
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "transport": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}}
        (args.report_dir / f"rank_{rank}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        dist.barrier()
        if rank == 0:
            for index in range(world):
                payload = json.loads((args.report_dir / f"rank_{index}.json").read_text())
                if payload["rank"] != index or payload["world_size"] != world:
                    raise ValueError("通信报告缺少 rank 或数量不一致")
            (args.report_dir / "complete.json").write_text(json.dumps(report), encoding="utf-8")
        print(f"rank={rank} DDP/backward/all_reduce/broadcast PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
