"""GeoGR 隔离入口：在纯 DDP 下兼容 PEFT 0.19 与 Transformers 4.46。"""

from __future__ import annotations

from geogr_full_pipeline.peft_compat import configure_ddp_peft_save
from geogr_full_pipeline.ddp_diagnostics import logged_ddp
from torch.distributed.elastic.multiprocessing.errors import record


@record
def main() -> None:
    configure_ddp_peft_save()
    from tap_sid import train_tap_sid as trainer

    # 只替换本隔离进程的 DDP 构造入口，公共 TAP 代码及构造参数不变。
    trainer.DistributedDataParallel = logged_ddp
    trainer.main()


if __name__ == "__main__":
    main()
