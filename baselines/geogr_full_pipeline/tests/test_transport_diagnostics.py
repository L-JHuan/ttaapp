"""回归轻量预检漏检模型加载后故障，以及长 torchrun 汇总遮挡根因。"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class TransportDiagnosticsTest(unittest.TestCase):
    def test_logged_ddp_preserves_isinstance_contract(self):
        from geogr_full_pipeline import ddp_diagnostics

        # 公共训练器的保存、修复和评估均用 isinstance 判断 DDP，入口必须仍是类型。
        self.assertTrue(issubclass(ddp_diagnostics.logged_ddp, ddp_diagnostics.TorchDDP))

    def test_default_compatibility_is_used_even_when_native_would_pass(self):
        command = (
            'set -Eeuo pipefail; source "$HELPER"\n'
            'run_stage() { echo "$*"; return 0; }\n'
            'check_gpu_communication\n'
            'echo "profile=${NCCL_P2P_DISABLE:-unset}/${NCCL_SHM_DISABLE:-unset} count=$GPU_COUNT"\n'
        )
        env = {**os.environ, "HELPER": str(ROOT / "scripts/pipeline_runtime.sh"),
               "GPUS": "0,1,2,3", "GPU_COUNT": "4", "PYTHON_BIN": sys.executable,
               "RUN_ROOT": "/tmp", "PIPELINE_RUN_ID": "test"}
        env.pop("GEOGR_NCCL_PROFILE", None)
        env.pop("NCCL_P2P_DISABLE", None)
        env.pop("NCCL_SHM_DISABLE", None)
        result = subprocess.run(["bash", "-c", command], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nccl_preflight_compat", result.stdout)
        self.assertIn("profile=1/1 count=4", result.stdout)
        self.assertEqual(result.stdout.count("--nproc_per_node=4"), 1)

    def test_first_backend_error_survives_long_launcher_summary(self):
        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            writer = root / "failure.py"
            writer.write_text(
                'import sys\n'
                'print("NCCL WARN Hggc failure: out of memory")\n'
                'print("[rank6]: Traceback (most recent call last):")\n'
                'print("[rank6]:   File trainer.py, line 1117, in main")\n'
                'print("[rank6]: torch.distributed.DistBackendError: NCCL internal error")\n'
                'for i in range(180): print("launcher summary rank", i)\n'
                'sys.exit(7)\n', encoding="utf-8",
            )
            result = subprocess.run(
                ["bash", "-c", 'set -Eeuo pipefail; source "$HELPER"; run_stage em_train "$PYTHON_BIN" "$WRITER"'],
                env={**os.environ, "HELPER": str(ROOT / "scripts/pipeline_runtime.sh"),
                     "RUN_ROOT": output, "PIPELINE_RUN_ID": "test", "PYTHON_BIN": sys.executable,
                     "WRITER": str(writer), "GEOGR_SCRIPT_ROOT": str(ROOT)},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 7)
            self.assertIn("Hggc failure: out of memory", result.stderr)
            self.assertIn("trainer.py, line 1117", result.stderr)
            self.assertLess(len(result.stderr.splitlines()), 140)

    def test_ddp_failure_records_memory_and_preserves_original_exception(self):
        from geogr_full_pipeline import ddp_diagnostics

        model = torch.nn.Linear(2, 2)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        error = RuntimeError("backend allocation failure")
        stream = io.StringIO()
        with patch.object(ddp_diagnostics.TorchDDP, "__init__", side_effect=error), contextlib.redirect_stdout(stream):
            with self.assertRaises(RuntimeError) as caught:
                ddp_diagnostics.logged_ddp(model)
        self.assertIs(caught.exception, error)
        for phase in ("before_ddp", "ddp_failed"):
            self.assertIn(phase, stream.getvalue())
        for key, value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_memory_api_failure_does_not_mask_backend_exception(self):
        from geogr_full_pipeline import ddp_diagnostics

        error = RuntimeError("original backend error")
        stream = io.StringIO()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "current_device", side_effect=RuntimeError("vendor API unavailable")), \
             patch.object(ddp_diagnostics.TorchDDP, "__init__", side_effect=error), \
             contextlib.redirect_stdout(stream):
            with self.assertRaises(RuntimeError) as caught:
                ddp_diagnostics.logged_ddp(torch.nn.Linear(2, 2))
        self.assertIs(caught.exception, error)
        self.assertIn("vendor API unavailable", stream.getvalue())

    def test_em_sft_and_cpt_all_use_diagnostics_and_elastic_record(self):
        for name in ("train_matched_sft.py", "train_cpt.py"):
            source = (ROOT / "geogr_full_pipeline" / name).read_text()
            self.assertIn("logged_ddp", source)
            self.assertIn("@record", source)


if __name__ == "__main__":
    unittest.main()
