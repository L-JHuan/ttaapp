"""只观察 GeoGR 的 DDP 初始化和内存，不修改模型或梯度同步规则。"""

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as TorchDDP


def memory_snapshot(phase, model):
    record = {
        "event": "GEOGR_DDP_MEMORY", "phase": phase,
        "time": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
        "host": socket.gethostname(), "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "stage": os.environ.get("GEOGR_STAGE", "unknown"),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "transport": {k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
    }
    try:
        parameters = list(model.parameters())
        record["parameter_bytes"] = sum(p.numel() * p.element_size() for p in parameters)
        record["trainable_parameter_bytes"] = sum(p.numel() * p.element_size() for p in parameters if p.requires_grad)
        record["parameter_devices"] = sorted({str(p.device) for p in parameters})
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            record["device"] = device
            record["device_name"] = torch.cuda.get_device_name(device)
            record["allocated_bytes"] = torch.cuda.memory_allocated(device)
            record["reserved_bytes"] = torch.cuda.memory_reserved(device)
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            free, total = torch.cuda.mem_get_info(device)
            record["device_free_bytes"] = free
            record["device_total_bytes"] = total
    except Exception as error:
        # 厂商兼容运行时可能不实现全部查询；不能用诊断错误遮挡原始故障。
        record["memory_query_error"] = f"{type(error).__name__}: {error}"
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    directory = os.environ.get("GEOGR_DIAGNOSTIC_ROOT")
    if directory:
        try:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            stage = Path(record["stage"]).name
            with (root / f"{stage}.rank_{record['rank']}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError as error:
            print(f"DDP diagnostic file unavailable: {error}", flush=True)


class LoggedDDP(TorchDDP):
    def __init__(self, model, *args, **kwargs):
        memory_snapshot("before_ddp", model)
        # 只释放未使用缓存，为通信库额外分配留下空间；保留活跃张量和 DDP 类型契约。
        if os.environ.get("GEOGR_DDP_EMPTY_CACHE", "1") == "1" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as error:
                print(f"DDP cache release unavailable: {error}", flush=True)
            memory_snapshot("after_empty_cache", model)
        try:
            super().__init__(model, *args, **kwargs)
        except Exception:
            memory_snapshot("ddp_failed", model)
            raise
        memory_snapshot("after_ddp", model)


logged_ddp = LoggedDDP
