"""回归训练完成后的通信组收尾及旧权重恢复保护。"""

import tempfile
import unittest
import json
import os
import re
import subprocess
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from geogr_full_pipeline import train_p2p_encoder as trainer
from geogr_full_pipeline import recover_p2p_export as recovery
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LlamaConfig, LlamaModel


def delayed_export_worker(rank, rendezvous, output):
    """用短超时模拟其他进程等待、主进程长时间导出的故障。"""
    import time

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=2),
    )
    trainer.finish_training_group(torch.nn.Linear(2, 2), True)
    if rank == 0:
        time.sleep(3)
    Path(output, f"rank{rank}.txt").write_text(str(dist.is_initialized()))


class P2PExportRecoveryTest(unittest.TestCase):
    def saved_fixture(self, output):
        root = Path(output)
        base = root / "base"
        base.mkdir()
        adapter = root / "p2p" / "encoder_adapter"
        model = get_peft_model(
            LlamaModel(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            )),
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=2, lora_alpha=4,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        model.save_pretrained(adapter)
        for name in ("tokenizer_config.json", "tokenizer.json"):
            (adapter / name).write_text("{}")
        inputs = {}
        for name in ("poi_info", "role_priors", "train_sequences"):
            path = root / f"{name}.csv"
            path.write_text(name)
            inputs[name] = path
        args = SimpleNamespace(
            **inputs, id_mappings=None, adapter_dir=adapter, encoder_model=base,
            max_length=128, dtype="bfloat16", allow_legacy_adapter=False,
        )
        return args, model

    def test_training_group_is_closed_before_export(self):
        model = type("Wrapped", (), {"module": object()})()
        events = []
        with patch.object(dist, "barrier", side_effect=lambda: events.append("barrier")), \
             patch.object(dist, "destroy_process_group", side_effect=lambda: events.append("destroy")):
            unwrapped = trainer.finish_training_group(model, True)
        self.assertIs(unwrapped, model.module)
        self.assertEqual(events, ["barrier", "destroy"])

    def test_single_process_does_not_use_collectives(self):
        model = object()
        with patch.object(dist, "barrier") as barrier, \
             patch.object(dist, "destroy_process_group") as destroy:
            self.assertIs(trainer.finish_training_group(model, False), model)
        barrier.assert_not_called()
        destroy.assert_not_called()

    def test_export_can_outlive_training_group_timeout(self):
        with tempfile.TemporaryDirectory() as output:
            rendezvous = "file://" + str(Path(output, "rendezvous").resolve())
            mp.spawn(delayed_export_worker, args=(rendezvous, output), nprocs=2, join=True)
            for rank in (0, 1):
                self.assertEqual(Path(output, f"rank{rank}.txt").read_text(), "False")

    def test_legacy_recovery_requires_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as output:
            args, model = self.saved_fixture(output)
            with self.assertRaisesRegex(ValueError, "旧 adapter"):
                recovery.inspect_saved_adapter(args)
            args.allow_legacy_adapter = True
            training, weights = recovery.inspect_saved_adapter(args)
            self.assertIsNone(training["global_step"])
            self.assertEqual(
                training["completion_evidence"],
                "legacy_adapter_allowed_without_training_completion_marker",
            )
            recovery.check_loaded_weights(model, weights)

    def test_recovery_rejects_missing_or_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as output:
            args, _ = self.saved_fixture(output)
            args.allow_legacy_adapter = True
            file = args.adapter_dir / "adapter_model.safetensors"
            file.unlink()
            with self.assertRaisesRegex(ValueError, "权重文件"):
                recovery.inspect_saved_adapter(args)
            file.write_bytes(b"broken")
            with self.assertRaises(Exception):
                recovery.inspect_saved_adapter(args)

    def test_completed_training_marker_checks_input_identity(self):
        with tempfile.TemporaryDirectory() as output:
            args, _ = self.saved_fixture(output)
            marker = args.adapter_dir.parent / "p2p_training_report.json"
            marker.write_text(json.dumps({
                "training_complete": True,
                "input_sha256": trainer.input_fingerprints(args),
                "encoder_model": str(args.encoder_model.resolve()),
                "max_length": 128, "dtype": "bfloat16", "encode_limit": 0,
            }))
            recovery.inspect_saved_adapter(args)
            args.poi_info.write_text("changed")
            with self.assertRaisesRegex(ValueError, "输入不一致"):
                recovery.inspect_saved_adapter(args)

    def test_weight_verification_rejects_partial_or_changed_adapter(self):
        with tempfile.TemporaryDirectory() as output:
            args, model = self.saved_fixture(output)
            args.allow_legacy_adapter = True
            _, weights = recovery.inspect_saved_adapter(args)
            key = next(iter(weights))
            partial = dict(weights)
            partial.pop(key)
            with self.assertRaisesRegex(ValueError, "参数键"):
                recovery.check_loaded_weights(model, partial)
            changed = dict(weights)
            changed[key] = weights[key] + 1
            with self.assertRaisesRegex(ValueError, "权重不一致"):
                recovery.check_loaded_weights(model, changed)

    def test_runner_uses_recovery_without_overwriting_training_log(self):
        path = Path(__file__).resolve().parents[1] / "scripts/run_industrial_pipeline.sh"
        script = path.read_text()
        self.assertIn("geogr_full_pipeline.recover_p2p_export", script)
        self.assertIn("P2P_RECOVER_LEGACY", script)
        self.assertIn("p2p_export_recovery_", script)
        self.assertIn('trap \'pipeline_failed "$?" "$LINENO"\' ERR', script)

    def test_runner_defaults_to_legacy_recovery_but_allows_disabling(self):
        """一键入口默认兼容旧权重，显式设为0时仍可禁用。"""
        path = Path(__file__).resolve().parents[1] / "scripts/run_industrial_pipeline.sh"
        assignment = re.search(r"^P2P_RECOVER_LEGACY=.*$", path.read_text(), re.MULTILINE)
        self.assertIsNotNone(assignment)
        command = assignment.group(0) + '\nprintf "%s" "$P2P_RECOVER_LEGACY"'
        for override, expected in ((None, "1"), ("0", "0")):
            env = dict(os.environ)
            env.pop("P2P_RECOVER_LEGACY", None)
            if override is not None:
                env["P2P_RECOVER_LEGACY"] = override
            result = subprocess.run(
                ["bash", "-c", command], env=env, check=True,
                capture_output=True, text=True,
            )
            self.assertEqual(result.stdout, expected)


if __name__ == "__main__":
    unittest.main()
