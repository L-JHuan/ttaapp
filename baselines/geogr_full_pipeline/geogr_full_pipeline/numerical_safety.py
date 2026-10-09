"""GeoGR 隔离数值保护：不改损失、优化器或主线训练循环。"""

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


class NonFiniteTrainingError(RuntimeError):
    """任一卡发现无效监督或非有限值，所有健康 rank 一致终止。"""


class TrainingGuard:
    def __init__(self, model, device, epoch, global_step):
        self.device = torch.device(device)
        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.epoch = epoch
        self.global_step = global_step
        self.micro_step = 0
        self.batch = {}
        self.parameters = [(name, value) for name, value in model.named_parameters()
                           if value.requires_grad]

    def check(self, phase, checks, names=None):
        # 先规约小型状态张量，再抛异常；不能让好卡继续进入下一次梯度同步。
        invalid = ~torch.stack(checks).all() if checks else torch.zeros(
            (), device=self.device, dtype=torch.bool)
        flags = torch.zeros(self.world_size, device=self.device, dtype=torch.int32)
        flags[self.rank] = invalid.to(torch.int32)
        if self.distributed:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        if not bool(flags.any().item()):
            return
        bad_ranks = flags.nonzero().flatten().cpu().tolist()
        local_bad = bool(flags[self.rank].item())
        failed_names = []
        if local_bad and names:
            failed_names = [name for name, check in zip(names, checks)
                            if not bool(check.item())][:8]
        report = {
            "event": "GEOGR_NONFINITE", "phase": phase,
            "time": datetime.now(timezone.utc).isoformat(),
            "host": socket.gethostname(), "pid": os.getpid(),
            "stage": os.environ.get("GEOGR_STAGE", "unknown"),
            "rank": self.rank, "world_size": self.world_size,
            "device": str(self.device), "epoch": self.epoch,
            "micro_step": self.micro_step, "global_step": self.global_step,
            "bad_ranks": bad_ranks, "local_bad": local_bad,
            "failed_tensors": failed_names,
        }
        # 仅记录形状和监督数量，不泄露 token、用户文本、位置或原始行为。
        if local_bad:
            for key in ("input_ids", "labels", "attention_mask"):
                value = self.batch.get(key)
                if isinstance(value, torch.Tensor):
                    report[f"{key}_shape" if key != "input_ids" else "batch_shape"] = list(value.shape)
            labels = self.batch.get("labels")
            if isinstance(labels, torch.Tensor) and labels.ndim == 2:
                report["supervised_tokens_per_sample"] = (labels[:, 1:] != -100).sum(1).cpu().tolist()
            attention = self.batch.get("attention_mask")
            if isinstance(attention, torch.Tensor) and attention.ndim == 2:
                report["sequence_lengths"] = attention.sum(1).cpu().tolist()
        line = json.dumps(report, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        directory = os.environ.get("GEOGR_DIAGNOSTIC_ROOT")
        if directory:
            try:
                root = Path(directory)
                root.mkdir(parents=True, exist_ok=True)
                stage = Path(report["stage"]).name
                with (root / f"{stage}.rank_{self.rank}.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as error:
                print(f"数值诊断文件写入失败：{error}", flush=True)
        raise NonFiniteTrainingError(
            f"GEOGR_NONFINITE phase={phase} bad_ranks={bad_ranks} "
            f"epoch={self.epoch} micro_step={self.micro_step} global_step={self.global_step}; "
            "本次训练停止，不保存成功标记或进入后续阶段"
        )

    def check_parameters(self, phase):
        self.check(phase, [torch.isfinite(value.detach()).all()
                           for _, value in self.parameters],
                   [name for name, _ in self.parameters])

    def before_forward(self, module, args, kwargs):
        self.micro_step += 1
        self.batch = kwargs
        labels = kwargs.get("labels")
        valid = (labels[:, 1:] != -100).any(1).all() if (
            isinstance(labels, torch.Tensor) and labels.ndim == 2
            and labels.shape[0] > 0 and labels.shape[1] > 1
        ) else torch.zeros((), device=self.device, dtype=torch.bool)
        self.check("labels", [valid], ["shifted_supervised_labels"])

    def after_forward(self, module, args, kwargs, output):
        values = {key: output[key] for key in ("loss", "lm_loss") if key in output} if isinstance(
            output, dict) else {"loss": output.loss}
        self.check("loss", [torch.isfinite(value.detach()).all() for value in values.values()],
                   list(values))

    def before_step(self, optimizer, args, kwargs):
        gradients = [(name, value.grad) for name, value in self.parameters if value.grad is not None]
        self.check("gradient", [torch.isfinite(value.detach()).all() for _, value in gradients],
                   [name for name, _ in gradients])

    def after_step(self, optimizer, args, kwargs):
        self.check_parameters("parameters_after_update")
        self.global_step += 1


@contextmanager
def guard_training(model, optimizer, device, epoch, global_step):
    guard = TrainingGuard(model, device, epoch, global_step)
    guard.check_parameters("parameters_before_training")
    handles = []
    try:
        handles.append(model.register_forward_pre_hook(guard.before_forward, with_kwargs=True))
        handles.append(model.register_forward_hook(guard.after_forward, with_kwargs=True))
        handles.append(optimizer.register_step_pre_hook(guard.before_step))
        handles.append(optimizer.register_step_post_hook(guard.after_step))
        yield guard
    finally:
        for handle in handles:
            handle.remove()


def install_training_guard(trainer):
    """仅在当前 GeoGR 进程包裹原循环，保留主线代码与正常更新语义。"""
    original = trainer.train_epoch
    if getattr(original, "__geogr_guarded__", False):
        return

    @wraps(original)
    def guarded(model, loader, optimizer, device, args, epoch, rank, distributed, global_step):
        with guard_training(model, optimizer, device, epoch, global_step):
            return original(model, loader, optimizer, device, args, epoch, rank, distributed, global_step)

    guarded.__geogr_guarded__ = True
    trainer.train_epoch = guarded
