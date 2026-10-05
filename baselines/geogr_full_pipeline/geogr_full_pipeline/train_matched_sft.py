"""GeoGR 隔离入口：在纯 DDP 下兼容 PEFT 0.19 与 Transformers 4.46。"""

from __future__ import annotations

import peft.utils.save_and_load as peft_save_and_load


def _skip_tensor_parallel_sharding(model, state_dict, adapter_name):
    """本实验仅使用 DDP；模型没有 HF tensor-parallel plan，无需 TP 权重分片。"""
    del model, state_dict, adapter_name


def main() -> None:
    peft_save_and_load._maybe_shard_state_dict_for_tp = _skip_tensor_parallel_sharding
    from tap_sid.train_tap_sid import main as tap_main

    tap_main()


if __name__ == "__main__":
    main()
