import json
import tempfile
import unittest
from pathlib import Path

from tap_sid.eval_sharding import merge_shards, split_rows


def sid(value: int) -> str:
    return f"<a_{value}><b_0><c_0><d_0><e_0>"


class EvalShardingTest(unittest.TestCase):
    def test_split_rows_covers_each_sample_once(self) -> None:
        rows = [{"output": sid(index)} for index in range(7)]
        shards = split_rows(rows, 3)

        self.assertEqual([len(shard) for shard in shards], [3, 2, 2])
        indices = sorted(row["_tap_sample_index"] for shard in shards for row in shard)
        self.assertEqual(indices, list(range(7)))

    def test_merge_restores_order_and_recomputes_metrics(self) -> None:
        dataset_rows = [{"output": sid(index)} for index in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "test.json"
            dataset.write_text(json.dumps(dataset_rows), encoding="utf-8")

            shard_predictions = [
                [
                    {"sample_index": 2, "predictions": [sid(2)]},
                    {"sample_index": 0, "predictions": [sid(0)]},
                ],
                [
                    {"sample_index": 3, "predictions": [sid(9), sid(3)]},
                    {"sample_index": 1, "predictions": [sid(9)]},
                ],
            ]
            for shard_index, rows in enumerate(shard_predictions):
                (root / f"predictions_{shard_index:05d}.json").write_text(
                    json.dumps(rows),
                    encoding="utf-8",
                )
                (root / f"metrics_{shard_index:05d}.json").write_text(
                    json.dumps(
                        {
                            "shard_index": shard_index,
                            "num_shards": 2,
                            "samples": len(rows),
                            "illegal_sid_predictions": 0,
                            "pred_sid_length_histogram": {"5": len(rows)},
                        }
                    ),
                    encoding="utf-8",
                )

            merged, metrics = merge_shards(dataset, root, 2)

        self.assertTrue(all("sample_index" not in row for row in merged))
        self.assertEqual([row["predictions"][0] for row in merged], [sid(0), sid(9), sid(2), sid(9)])
        self.assertEqual(metrics["recall@1"], 0.5)
        self.assertEqual(metrics["recall@5"], 0.75)
        self.assertEqual(metrics["recall@10"], 0.75)
        self.assertTrue(metrics["multi_gpu_eval"])
        self.assertTrue(metrics["sample_coverage_checked"])

    def test_merge_rejects_missing_samples(self) -> None:
        dataset_rows = [{"output": sid(index)} for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "test.json"
            dataset.write_text(json.dumps(dataset_rows), encoding="utf-8")
            (root / "predictions_00000.json").write_text(
                json.dumps([{"sample_index": 0, "predictions": [sid(0)]}]),
                encoding="utf-8",
            )
            (root / "metrics_00000.json").write_text(
                json.dumps({"shard_index": 0, "num_shards": 1, "samples": 1}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "覆盖不完整"):
                merge_shards(dataset, root, 1)


if __name__ == "__main__":
    unittest.main()
