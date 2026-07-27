import unittest

import pandas as pd

from gnpr_baseline.prepare_realworld_data import assign_split, collapse_consecutive_same_poi


class CollapseConsecutiveSamePoiTest(unittest.TestCase):
    def test_keeps_first_report_of_each_consecutive_poi_state(self) -> None:
        events = pd.DataFrame(
            [
                {"_user": "u1", "_poi": "a", "_time": pd.Timestamp("2026-07-01 09:05:00Z")},
                {"_user": "u2", "_poi": "c", "_time": pd.Timestamp("2026-07-01 09:02:00Z")},
                {"_user": "u1", "_poi": "a", "_time": pd.Timestamp("2026-07-01 09:00:00Z")},
                {"_user": "u1", "_poi": "b", "_time": pd.Timestamp("2026-07-01 10:00:00Z")},
                {"_user": "u2", "_poi": "c", "_time": pd.Timestamp("2026-07-01 09:10:00Z")},
                {"_user": "u1", "_poi": "b", "_time": pd.Timestamp("2026-07-01 10:30:00Z")},
                {"_user": "u1", "_poi": "a", "_time": pd.Timestamp("2026-07-01 11:00:00Z")},
            ]
        )

        collapsed, removed = collapse_consecutive_same_poi(events)

        self.assertEqual(removed, 3)
        self.assertEqual(
            list(collapsed[["_user", "_poi"]].itertuples(index=False, name=None)),
            [("u1", "a"), ("u1", "b"), ("u1", "a"), ("u2", "c")],
        )
        self.assertEqual(
            collapsed.iloc[0]["_time"],
            pd.Timestamp("2026-07-01 09:00:00Z"),
        )


class AssignSplitTest(unittest.TestCase):
    def test_three_way_time_split(self) -> None:
        train_end = pd.Timestamp("2026-07-14 23:59:59Z")
        validation_end = pd.Timestamp("2026-07-17 23:59:59Z")

        self.assertEqual(assign_split(pd.Timestamp("2026-07-14 12:00:00Z"), train_end, validation_end), "train")
        self.assertEqual(
            assign_split(pd.Timestamp("2026-07-16 12:00:00Z"), train_end, validation_end),
            "validation",
        )
        self.assertEqual(assign_split(pd.Timestamp("2026-07-18 12:00:00Z"), train_end, validation_end), "test")

    def test_no_validation_sends_all_later_targets_to_test(self) -> None:
        train_end = pd.Timestamp("2026-07-17 23:59:59Z")

        self.assertEqual(assign_split(pd.Timestamp("2026-07-17 12:00:00Z"), train_end, None), "train")
        self.assertEqual(assign_split(pd.Timestamp("2026-07-18 00:00:00Z"), train_end, None), "test")


if __name__ == "__main__":
    unittest.main()
