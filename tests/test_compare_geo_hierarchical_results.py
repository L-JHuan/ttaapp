import json
import tempfile
import unittest
from pathlib import Path

from baselines.compare_geo_hierarchical_results import build_comparison


class GeoHierarchicalComparisonTest(unittest.TestCase):
    def test_builds_absolute_and_relative_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tap = root / "tap.json"
            baseline = root / "baseline.json"
            tap.write_text(
                json.dumps(
                    {
                        "recall@1": 0.4,
                        "recall@5": 0.6,
                        "recall@10": 0.7,
                        "ndcg@10": 0.5,
                        "samples": 100,
                    }
                ),
                encoding="utf-8",
            )
            baseline.write_text(
                json.dumps(
                    {
                        "recall@1": 0.2,
                        "recall@5": 0.5,
                        "recall@10": 0.5,
                        "ndcg@10": 0.4,
                        "samples": 100,
                    }
                ),
                encoding="utf-8",
            )

            rows = build_comparison(
                {"TAP-SID": tap, "Spacetime-GR adapted": baseline},
                reference="TAP-SID",
            )

        by_method = {row["method"]: row for row in rows}
        compared = by_method["Spacetime-GR adapted"]
        self.assertAlmostEqual(compared["delta_recall@1"], -0.2)
        self.assertAlmostEqual(compared["relative_recall@1_pct"], -50.0)
        self.assertEqual(compared["samples"], 100)

    def test_rejects_mismatched_sample_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.json"
            second = root / "second.json"
            payload = {
                "recall@1": 0.4,
                "recall@5": 0.6,
                "recall@10": 0.7,
                "ndcg@10": 0.5,
                "samples": 100,
            }
            first.write_text(json.dumps(payload), encoding="utf-8")
            payload["samples"] = 99
            second.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "样本数不一致"):
                build_comparison({"TAP-SID": first, "GeoGR adapted": second}, "TAP-SID")


if __name__ == "__main__":
    unittest.main()
