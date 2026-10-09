"""GeoGR 隔离入口：在纯 DDP 下兼容 PEFT 0.19 与 Transformers 4.46。"""

from __future__ import annotations

from geogr_full_pipeline.peft_compat import configure_ddp_peft_save


def main() -> None:
    configure_ddp_peft_save()
    from tap_sid.train_tap_sid import main as tap_main

    tap_main()


if __name__ == "__main__":
    main()
