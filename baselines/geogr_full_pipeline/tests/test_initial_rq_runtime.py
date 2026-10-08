"""验证初始 RQ 的线程隔离、失败日志及已完成 P2P 的复用。"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


THREAD_VALUES = {
    "OPENBLAS_NUM_THREADS": "128",
    "OPENBLAS_DEFAULT_NUM_THREADS": "128",
    "OMP_NUM_THREADS": "96",
    "MKL_NUM_THREADS": "64",
    "PYTHONFAULTHANDLER": "0",
}


class InitialRqRuntimeTest(unittest.TestCase):
    def run_stage(self, output, *, fail=False, complete=False):
        root = Path(output)
        run = root / "run"
        for name in ("p2p", "sid", "logs"):
            (run / name).mkdir(parents=True)
        (run / "p2p/p2p_encoder_report.json").write_text("{}")
        (run / "p2p/refined_embeddings.npz").write_bytes(b"completed_p2p")
        old_log = run / "logs/initial_rq.log"
        old_log.write_text("previous_failed_rq_log")
        if complete:
            (run / "sid/initial_rq_sid.csv").write_text("completed_sid")
            (run / "sid/initial_rq_report.json").write_text("{}")
        record = root / "calls.jsonl"
        probe = root / "probe.py"
        probe.write_text(
            '#!' + sys.executable + '\nimport json, os, sys\n'
            'from pathlib import Path\n'
            'args = sys.argv[1:]\n'
            'keys = ' + repr(list(THREAD_VALUES)) + '\n'
            'with Path(os.environ["CALL_RECORD"]).open("a") as stream:\n'
            '    stream.write(json.dumps({"args": args, "env": {key: os.environ.get(key) for key in keys}}) + "\\n")\n'
            'if args == ["--probe_after"]:\n'
            '    sys.exit(0)\n'
            'if "geogr_full_pipeline.build_initial_rq_sid" not in args:\n'
            '    raise RuntimeError("unexpected training or recovery call")\n'
            'if os.environ.get("FAIL_RQ") == "1":\n'
            '    print("rq_native_crash_marker", file=sys.stderr, flush=True)\n'
            '    sys.exit(139)\n'
            'Path(args[args.index("--output_csv") + 1]).write_text("completed_sid")\n'
            'Path(args[args.index("--report_json") + 1]).write_text("{}")\n'
        )
        probe.chmod(0o755)
        runner = Path(__file__).resolve().parents[1] / "scripts/run_industrial_pipeline.sh"
        script = runner.read_text()
        error_handler = script.split('CURRENT_STAGE_LOG=""', 1)[1].split("json_ok()", 1)[0]
        stage = script.split('P2P_ROOT="$RUN_ROOT/p2p"', 1)[1].split('CURRENT_SID="$INITIAL_SID"', 1)[0]
        command = (
            'set -Eeuo pipefail\nCURRENT_STAGE_LOG=""\n'
            'PYTHON_BIN="$PROBE_FILE"\n'
            'timestamp() { date; }\nlog() { printf "%s\\n" "$*"; }\n'
            'json_ok() { "$REAL_PYTHON" -m json.tool "$1" >/dev/null; }\n'
            'require_new_dir() { return 99; }\n'
            + error_handler + '\nP2P_ROOT="$RUN_ROOT/p2p"\n'
            + stage + '\n"$PYTHON_BIN" --probe_after\n'
        )
        env = {
            **os.environ, **THREAD_VALUES,
            "RUN_ROOT": str(run), "REAL_PYTHON": sys.executable,
            "PROBE_FILE": str(probe), "CALL_RECORD": str(record),
            "CODEBOOK_SIZE": "8", "FAIL_RQ": "1" if fail else "0",
        }
        result = subprocess.run(
            ["bash", "-c", command], env=env, capture_output=True, text=True,
        )
        calls = [json.loads(line) for line in record.read_text().splitlines()]
        return result, calls, run

    def test_thread_limits_are_local_and_completed_p2p_is_reused(self):
        with tempfile.TemporaryDirectory() as output:
            result, calls, run = self.run_stage(output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["env"], {key: "1" for key in THREAD_VALUES})
            self.assertEqual(calls[1]["env"], THREAD_VALUES)
            self.assertEqual((run / "p2p/refined_embeddings.npz").read_bytes(), b"completed_p2p")
            self.assertEqual((run / "logs/initial_rq.log").read_text(), "previous_failed_rq_log")
            self.assertEqual(len(list((run / "logs").glob("initial_rq_*.log"))), 1)

    def test_rq_failure_stops_pipeline_and_reports_stage_log(self):
        with tempfile.TemporaryDirectory() as output:
            result, calls, run = self.run_stage(output, fail=True)
            self.assertEqual(result.returncode, 139)
            self.assertEqual(len(calls), 1)
            self.assertIn("rq_native_crash_marker", result.stderr)
            self.assertIn("Stage log:", result.stderr)
            self.assertEqual((run / "logs/initial_rq.log").read_text(), "previous_failed_rq_log")

    def test_completed_initial_rq_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as output:
            result, calls, run = self.run_stage(output, complete=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([call["args"] for call in calls], [["--probe_after"]])
            self.assertEqual((run / "sid/initial_rq_sid.csv").read_text(), "completed_sid")

    def test_real_cpu_rq_runs_with_single_thread_backends(self):
        """小数据实际运行三层 K-means，不用模拟结果替代算法检查。"""
        import numpy as np

        with tempfile.TemporaryDirectory() as output:
            root = Path(output)
            embeddings = root / "embeddings.npz"
            np.savez_compressed(
                embeddings, pids=np.arange(96),
                embeddings=np.random.default_rng(2024).normal(size=(96, 32)).astype(np.float32),
            )
            program = (
                "import json; from geogr_full_pipeline.build_initial_rq_sid import main; "
                "from threadpoolctl import threadpool_info; main(); "
                "print(json.dumps(threadpool_info()))"
            )
            env = {**os.environ, **{key: "1" for key in THREAD_VALUES}}
            result = subprocess.run(
                [sys.executable, "-c", program, "--embeddings_npz", str(embeddings),
                 "--output_csv", str(root / "sid.csv"), "--report_json", str(root / "report.json"),
                 "--codebook_size", "8", "--seed", "2024"],
                env=env, capture_output=True, text=True, check=True,
            )
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["status"], "GEOGR_FULL_INITIAL_RQ_OK")
            self.assertEqual(report["catalog_pois"], 96)
            self.assertEqual(report["num_layers"], 3)
            pools = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertTrue(pools)
            self.assertTrue(all(pool["num_threads"] == 1 for pool in pools), pools)


if __name__ == "__main__":
    unittest.main()
