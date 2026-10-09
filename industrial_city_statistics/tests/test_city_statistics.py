from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from industrial_city_statistics.analyze_cities import analyze, main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CityStatisticsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tap, self.gnpr = self.root / "tap", self.root / "gnpr"
        self.processed = self.root / "processed"
        self.metadata = self.root / "pois.jsonl"
        self.output = self.root / "result"
        # 测试数据特意使用不同的原始/内部编号，以及相反的方法样本顺序。
        write_json(self.processed / "id_mappings.json", {
            "poi_id_to_internal": {"raw-a": 0, "raw-b": 1, "raw-c": 2}})
        self.metadata.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in [
            {"poi编号": "raw-a", "区县编码": "A", "区县中文": "甲市"},
            {"poi编号": "raw-b", "区县编码": "B", "区县中文": "乙市"},
            {"poi编号": "raw-c", "区县编码": "C", "区县中文": "丙市"},
            {"poi编号": "extra", "区县编码": "D", "区县中文": "丁市"},
        ]), encoding="utf-8")
        for root, letter, reverse in [(self.tap, "a", False), (self.gnpr, "b", True)]:
            codes = [f"<{letter}_{i}>" for i in range(3)]
            write_csv(root / "codebook" / ("tap_sid.csv" if root == self.tap else "gnpr_sid.csv"),
                      [{"pid": i, "sid_tokens": sid} for i, sid in enumerate(codes)])
            data, predictions = [], []
            for i in range(2):
                text = f"User_7 checkin history: t visited {codes[2]}.\nWhen t{i} user_7 is likely to visit:"
                data.append({"input": text, "output": codes[i]})
                ranked = [codes[i], codes[2]] if root == self.tap else [codes[2], codes[i]]
                predictions.append({"input": text, "gold": codes[i], "predictions": ranked})
            if reverse:
                data.reverse()
                predictions.reverse()
            for i, row in enumerate(predictions):
                row["sample_index"] = i
            write_json(root / "data/llm_test.json", data)
            write_json(root / "eval/test_predictions.json", predictions)
            write_json(root / "eval/test_metrics.json", {
                "recall@1": 1.0 if root == self.tap else 0.0,
                "recall@5": 1.0, "recall@10": 1.0,
                "ndcg@10": 1.0 if root == self.tap else 1 / math.log2(3),
                "samples": 2, "prediction_samples": 2,
                "legal_sid_count": 3, "k": 10,
            })

    def run_analysis(self):
        return analyze(self.tap, self.gnpr, self.processed, self.output,
                       poi_city_jsonl=self.metadata)

    def change_json(self, path, edit):
        value = json.loads(path.read_text(encoding="utf-8"))
        edit(value)
        write_json(path, value)

    def test_aligns_different_order_counts_users_and_no_test_cities(self):
        report = self.run_analysis()
        self.assertEqual(report["status"], "PASS")
        summary = json.loads((self.output / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["test_users"], 1)
        self.assertEqual(summary["sum_city_test_users"], 2)
        self.assertEqual(summary["catalog_pois"], 3)
        self.assertEqual(summary["catalog_cities"], 3)
        self.assertEqual(summary["test_cities"], 2)
        cities = json.loads((self.output / "city_statistics.json").read_text(encoding="utf-8"))
        self.assertEqual(len(cities), 3)
        city = next(row for row in cities if row["city_code"] == "C")
        self.assertEqual(city["test_samples"], 0)
        self.assertIsNone(city["tap_R1"])
        self.assertIsNone(cities[0]["relative_R1_pct"])
        self.assertEqual(cities[0]["delta_R1"], 1)
        self.assertNotIn("raw-a", (self.output / "city_statistics.json").read_text(encoding="utf-8"))

    def test_rejects_catalog_mismatch(self):
        path = self.gnpr / "codebook/gnpr_sid.csv"
        write_csv(path, [{"pid": i, "sid_tokens": f"<b_{i}>"} for i in range(4)])
        self.change_json(self.gnpr / "eval/test_metrics.json", lambda v: v.update({"legal_sid_count": 4}))
        with self.assertRaisesRegex(ValueError, "目录"):
            self.run_analysis()

    def test_rejects_wrong_official_metric(self):
        self.change_json(self.tap / "eval/test_metrics.json", lambda v: v.update({"recall@1": 0.1}))
        with self.assertRaisesRegex(ValueError, "指标"):
            self.run_analysis()

    def test_rejects_missing_city_mapping(self):
        self.metadata.write_text('{"poi编号":"extra","区县编码":"D","区县中文":"丁市"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "城市"):
            self.run_analysis()

    def test_rejects_stale_prediction_input(self):
        self.change_json(self.tap / "eval/test_predictions.json", lambda v: v[0].update({"input": "wrong"}))
        with self.assertRaisesRegex(ValueError, "input"):
            self.run_analysis()

    def test_rejects_duplicate_sample_index(self):
        self.change_json(self.tap / "eval/test_predictions.json", lambda v: v[1].update({"sample_index": 0}))
        with self.assertRaisesRegex(ValueError, "sample_index"):
            self.run_analysis()

    def test_rejects_mismatched_history(self):
        for name in ["data/llm_test.json", "eval/test_predictions.json"]:
            self.change_json(self.gnpr / name, lambda v: v[0].update({"input": v[0]["input"].replace("visited <b_2>", "visited <b_0>")}))
        with self.assertRaisesRegex(ValueError, "样本"):
            self.run_analysis()

    def test_reads_spark_json_directory(self):
        for root in [self.tap, self.gnpr]:
            path = root / "data/llm_test.json"
            rows = json.loads(path.read_text(encoding="utf-8"))
            path.unlink()
            directory = root / "data/llm_test.jsonl"
            directory.mkdir()
            (directory / "_SUCCESS").touch()
            for i, row in enumerate(rows):
                write_json(directory / f"part-{i:05d}.json", row)
        self.assertEqual(self.run_analysis()["status"], "PASS")

    def test_does_not_overwrite_outputs(self):
        self.run_analysis()
        with self.assertRaises(FileExistsError):
            self.run_analysis()

    def test_success_cli_prints_only_output_location(self):
        args = ["--tap-run-root", str(self.tap), "--gnpr-run-root", str(self.gnpr),
                "--processed-root", str(self.processed), "--poi-city-jsonl", str(self.metadata),
                "--output-dir", str(self.output)]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            self.assertEqual(main(args), 0)
        self.assertEqual(len(stream.getvalue().strip().splitlines()), 1)
        self.assertNotIn("甲市", stream.getvalue())

    def test_metadata_duplicate_is_rejected(self):
        first = self.metadata.read_text(encoding="utf-8").splitlines()[0]
        with self.metadata.open("a", encoding="utf-8") as handle:
            handle.write("\n" + first)
        with self.assertRaisesRegex(ValueError, "重复"):
            self.run_analysis()

    def test_sample_count_is_checked(self):
        self.change_json(self.tap / "eval/test_metrics.json", lambda v: v.update({"samples": 100}))
        with self.assertRaisesRegex(ValueError, "样本数"):
            self.run_analysis()

    def test_boundary_mode_keeps_unmatched_city(self):
        write_csv(self.processed / "poi_info.csv", [
            {"pid": 0, "longitude": 116.4, "latitude": 39.9},
            {"pid": 1, "longitude": 121.47, "latitude": 31.23},
            {"pid": 2, "longitude": 0, "latitude": 0},
        ])
        report = analyze(self.tap, self.gnpr, self.processed, self.output)
        self.assertEqual(report["city_mapping"]["unmatched_catalog_pois"], 1)
        rows = json.loads((self.output / "city_statistics.json").read_text(encoding="utf-8"))
        self.assertEqual({r["city_code"] for r in rows}, {"110000", "310000", "UNMATCHED"})

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "本环境没有pyarrow")
    def test_reads_existing_spark_poi_mapping_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        (self.processed / "id_mappings.json").unlink()
        directory = self.processed / "mappings/pois"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([
            {"_poi": "raw-a", "pid": 0}, {"_poi": "raw-b", "pid": 1},
            {"_poi": "raw-c", "pid": 2}]), directory / "part-00000.parquet")
        (directory / "_SUCCESS").touch()
        self.assertEqual(self.run_analysis()["status"], "PASS")

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "本环境没有pyarrow")
    def test_reads_existing_spark_catalog_parquet(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        directory = self.processed / "metadata/catalog"
        directory.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([
            {"pid": 0, "longitude": 116.4, "latitude": 39.9},
            {"pid": 1, "longitude": 121.47, "latitude": 31.23},
            {"pid": 2, "longitude": 0., "latitude": 0.}]), directory / "part-00000.parquet")
        (directory / "_SUCCESS").touch()
        report = analyze(self.tap, self.gnpr, self.processed, self.output)
        self.assertEqual(report["city_mapping"]["unmatched_catalog_pois"], 1)

    @unittest.skipUnless(os.name == "posix", "一键bash入口在Linux验证")
    def test_shell_wrapper_writes_files_without_table_on_terminal(self):
        config = self.root / "statistics.local.env"
        config.write_text("\n".join([
            f"TAP_RUN_ROOT='{self.tap}'", f"GNPR_RUN_ROOT='{self.gnpr}'",
            f"PROCESSED_ROOT='{self.processed}'", f"POI_CITY_JSONL='{self.metadata}'",
            f"CITY_STATS_ROOT='{self.output}'", f"PYTHON_BIN='{os.sys.executable}'",
        ]), encoding="utf-8")
        script = Path(__file__).resolve().parents[1] / "run_statistics.sh"
        result = subprocess.run(["bash", str(script), str(config)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("甲市", result.stdout)
        self.assertEqual(len(result.stdout.strip().splitlines()), 1)
        reports = list(self.output.glob("*/results/validation_report.json"))
        self.assertEqual(len(reports), 1)
        self.assertEqual(json.loads(reports[0].read_text(encoding="utf-8"))["status"], "PASS")

    def test_cli_writes_failure_report_not_statistics_to_terminal(self):
        self.change_json(self.tap / "eval/test_metrics.json", lambda v: v.update({"recall@1": float("nan")}))
        args = ["--tap-run-root", str(self.tap), "--gnpr-run-root", str(self.gnpr),
                "--processed-root", str(self.processed), "--poi-city-jsonl", str(self.metadata),
                "--output-dir", str(self.output)]
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            code = main(args)
        self.assertEqual(code, 1)
        self.assertNotIn("recall@1", stream.getvalue())
        self.assertEqual(json.loads((self.output / "validation_report.json").read_text(encoding="utf-8"))["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
