"""回归阶段失败可见性、续跑检查和纯 DDP 权重保存兼容。"""

import json
import os
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineObservabilityTest(unittest.TestCase):
    def test_failed_stage_has_log_and_stops_next_stage(self):
        with tempfile.TemporaryDirectory() as output:
            command = (
                'set -Eeuo pipefail\n'
                'source "$HELPER"\n'
                'run_stage em_train bash -c "echo native_failure >&2; exit 7"\n'
                'touch "$RUN_ROOT/should_not_exist"\n'
            )
            env = {**os.environ, "RUN_ROOT": output, "PIPELINE_RUN_ID": "test",
                   "HELPER": str(ROOT / "scripts/pipeline_runtime.sh")}
            result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 7, result.stderr)
            log = Path(output, "logs/em_train_test.log").read_text()
            self.assertIn("START", log)
            self.assertIn("native_failure", log)
            self.assertIn("FAILED exit=7", log)
            self.assertFalse(Path(output, "should_not_exist").exists())

    def test_successful_silent_stage_is_not_an_empty_log(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                ["bash", "-c", 'set -Eeuo pipefail; source "$HELPER"; run_stage rq true'],
                env={**os.environ, "RUN_ROOT": output, "PIPELINE_RUN_ID": "test",
                     "HELPER": str(ROOT / "scripts/pipeline_runtime.sh")},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("DONE exit=0", Path(output, "logs/rq_test.log").read_text())

    def test_rq_logs_real_layer_progress(self):
        source = (ROOT / "geogr_full_pipeline/sid_utils.py").read_text()
        self.assertIn("RQ layer", source)

    def test_cpt_uses_ddp_save_compatibility(self):
        source = (ROOT / "geogr_full_pipeline/train_cpt.py").read_text()
        self.assertIn("configure_ddp_peft_save", source)

    def test_preflight_before_first_training_and_unique_eval_retry(self):
        source = (ROOT / "scripts/run_industrial_pipeline.sh").read_text()
        self.assertLess(source.index("check_gpu_communication"), source.index('P2P_ROOT="$RUN_ROOT/p2p"'))
        self.assertIn('EVAL_RUN_ID="${run_id}_${PIPELINE_RUN_ID}"', source)
        self.assertIn("checkpoint_ready", source)

    def test_incomplete_checkpoint_is_preserved_not_deleted(self):
        from geogr_full_pipeline.runtime_checks import archive_incomplete_checkpoint

        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            checkpoint = root / "em/checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "partial.bin").write_bytes(b"saved_training")
            archived = archive_incomplete_checkpoint(checkpoint, root, "retry1")
            self.assertEqual((archived / "partial.bin").read_bytes(), b"saved_training")
            self.assertFalse(checkpoint.exists())
            with self.assertRaises(ValueError):
                archive_incomplete_checkpoint(root.parent, root, "retry1")

    def test_checkpoint_config_alone_is_not_completion(self):
        from geogr_full_pipeline.runtime_checks import checkpoint_ready

        with tempfile.TemporaryDirectory() as output:
            checkpoint = Path(output)
            adapter = checkpoint / "final_cpt"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
            self.assertFalse(checkpoint_ready(checkpoint, "final_cpt"))

    def test_native_failure_uses_same_gpu_count_in_compatibility_retry(self):
        command = (
            'set -Eeuo pipefail; source "$HELPER"\n'
            'run_stage() { echo "$*"; [[ "$1" == nccl_preflight_compat ]]; }\n'
            'check_gpu_communication\n'
            'echo "profile=$NCCL_P2P_DISABLE/$NCCL_SHM_DISABLE gpus=$GPUS count=$GPU_COUNT"\n'
        )
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True,
                                env={**os.environ, "HELPER": str(ROOT / "scripts/pipeline_runtime.sh"),
                                     "GPUS": "0,1,2,3", "GPU_COUNT": "4", "PYTHON_BIN": sys.executable,
                                     "RUN_ROOT": "/tmp", "PIPELINE_RUN_ID": "test"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profile=1/1 gpus=0,1,2,3 count=4", result.stdout)
        self.assertEqual(result.stdout.count("--nproc_per_node=4"), 2)

    def test_both_preflight_failures_stop_training(self):
        result = subprocess.run(
            ["bash", "-c", 'set -Eeuo pipefail; source "$HELPER"; run_stage() { return 7; }; check_gpu_communication; echo forbidden_training'],
            text=True, capture_output=True,
            env={**os.environ, "HELPER": str(ROOT / "scripts/pipeline_runtime.sh"),
                 "GPUS": "0,1", "GPU_COUNT": "2", "PYTHON_BIN": sys.executable,
                 "RUN_ROOT": "/tmp", "PIPELINE_RUN_ID": "test"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("forbidden_training", result.stdout)

    def test_real_two_process_preflight_on_cpu(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                 "-m", "geogr_full_pipeline.nccl_preflight", "--backend", "gloo", "--report_dir", output],
                text=True, capture_output=True, timeout=60,
                env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(Path(output, "complete.json").exists())
            self.assertEqual(len(list(Path(output).glob("rank_*.json"))), 2)

    def test_silent_long_stage_has_periodic_heartbeat(self):
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                ["bash", "-c", 'set -Eeuo pipefail; source "$HELPER"; run_stage quiet sleep 2'],
                env={**os.environ, "RUN_ROOT": output, "PIPELINE_RUN_ID": "test",
                     "STAGE_HEARTBEAT_SECONDS": "0.2",
                     "HELPER": str(ROOT / "scripts/pipeline_runtime.sh")},
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("RUNNING quiet", Path(output, "logs/quiet_test.log").read_text())


if __name__ == "__main__":
    unittest.main()
