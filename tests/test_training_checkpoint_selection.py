import unittest

from tap_sid.train_tap_sid import DistributedEvalSampler, select_best_epoch


class SelectBestEpochTest(unittest.TestCase):
    def test_distributed_validation_sampler_has_exact_coverage(self) -> None:
        dataset = list(range(10))
        shards = [
            list(DistributedEvalSampler(dataset, rank=rank, world_size=3))
            for rank in range(3)
        ]

        merged = [index for shard in shards for index in shard]
        self.assertEqual(sorted(merged), list(range(10)))
        self.assertEqual(len(merged), len(set(merged)))

    def test_selects_lowest_validation_lm_loss(self) -> None:
        summaries = [
            {"epoch": 1, "valid": {"lm_loss": 0.42}},
            {"epoch": 2, "valid": {"lm_loss": 0.31}},
            {"epoch": 3, "valid": {"lm_loss": 0.36}},
        ]

        selected = select_best_epoch(summaries)

        self.assertEqual(selected["epoch"], 2)
        self.assertEqual(selected["metric"], "lm_loss")
        self.assertEqual(selected["value"], 0.31)
        self.assertEqual(selected["mode"], "min")

    def test_validation_tie_keeps_earlier_epoch(self) -> None:
        summaries = [
            {"epoch": 1, "valid": {"lm_loss": 0.31}},
            {"epoch": 2, "valid": {"lm_loss": 0.31}},
        ]

        selected = select_best_epoch(summaries)

        self.assertEqual(selected["epoch"], 1)

    def test_without_validation_selects_final_epoch(self) -> None:
        summaries = [
            {"epoch": 1, "train": {"lm_loss": 0.42}},
            {"epoch": 2, "train": {"lm_loss": 0.31}},
            {"epoch": 3, "train": {"lm_loss": 0.36}},
        ]

        selected = select_best_epoch(summaries)

        self.assertEqual(selected["epoch"], 3)
        self.assertIsNone(selected["metric"])
        self.assertEqual(selected["mode"], "final")


if __name__ == "__main__":
    unittest.main()
