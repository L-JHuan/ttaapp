"""仅在未使用 tensor parallel 的 GeoGR DDP 入口中兼容 PEFT 保存。"""


def configure_ddp_peft_save():
    import peft.utils.save_and_load as save_and_load

    def skip_tp(model, state_dict, adapter_name):
        if getattr(model, "_tp_plan", None) or getattr(getattr(model, "base_model", None), "_tp_plan", None):
            raise ValueError("此兼容入口仅支持纯 DDP，不支持 tensor parallel")

    save_and_load._maybe_shard_state_dict_for_tp = skip_tp
