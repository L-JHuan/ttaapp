import unittest

from spacetime_gr.build_spacetime_sid import Poi, assign_hierarchical_codes


class SpacetimeSidBuilderTest(unittest.TestCase):
    def test_assigns_inner_ids_within_each_block(self):
        pois = [
            Poi(pid=20, latitude=40.7500, longitude=-73.9900),
            Poi(pid=10, latitude=40.7510, longitude=-73.9910),
            Poi(pid=30, latitude=40.8500, longitude=-73.9900),
        ]

        rows, report = assign_hierarchical_codes(
            pois,
            block_size_km=5.0,
            reference_latitude=40.75,
        )

        by_pid = {row.pid: row for row in rows}
        self.assertEqual(by_pid[10].block_id, by_pid[20].block_id)
        self.assertEqual((by_pid[10].inner_id, by_pid[20].inner_id), (1, 2))
        self.assertNotEqual(by_pid[20].block_id, by_pid[30].block_id)
        self.assertEqual(report["num_pois"], 3)
        self.assertEqual(report["num_blocks"], 2)

    def test_mapping_is_unique_and_input_order_invariant(self):
        pois = [
            Poi(pid=3, latitude=35.6800, longitude=139.7600),
            Poi(pid=1, latitude=35.6810, longitude=139.7610),
            Poi(pid=2, latitude=35.7800, longitude=139.7600),
        ]

        first, _ = assign_hierarchical_codes(
            pois,
            block_size_km=5.0,
            reference_latitude=35.7,
        )
        second, _ = assign_hierarchical_codes(
            list(reversed(pois)),
            block_size_km=5.0,
            reference_latitude=35.7,
        )

        first_map = {row.pid: row.sid for row in first}
        second_map = {row.pid: row.sid for row in second}
        self.assertEqual(first_map, second_map)
        self.assertEqual(len(set(first_map.values())), len(pois))

    def test_rejects_duplicate_pids(self):
        pois = [
            Poi(pid=1, latitude=40.0, longitude=-74.0),
            Poi(pid=1, latitude=40.1, longitude=-74.1),
        ]

        with self.assertRaisesRegex(ValueError, "重复 pid"):
            assign_hierarchical_codes(pois, block_size_km=5.0)


if __name__ == "__main__":
    unittest.main()
