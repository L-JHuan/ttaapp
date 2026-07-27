import unittest

from gnpr_baseline.prepare_industrial_spark import resolve_split_boundaries


class SparkSplitBoundariesTest(unittest.TestCase):
    def test_accepts_three_way_time_split(self) -> None:
        train_end, validation_end = resolve_split_boundaries(
            "2026-07-14T15:59:59Z",
            "2026-07-17T15:59:59Z",
        )

        self.assertLess(train_end, validation_end)

    def test_rejects_validation_before_train(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation_end"):
            resolve_split_boundaries(
                "2026-07-17T15:59:59Z",
                "2026-07-14T15:59:59Z",
            )

    def test_allows_train_test_only_split(self) -> None:
        train_end, validation_end = resolve_split_boundaries(
            "2026-07-17T15:59:59Z",
            "",
        )

        self.assertIsNotNone(train_end)
        self.assertIsNone(validation_end)


if __name__ == "__main__":
    unittest.main()
