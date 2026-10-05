import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from geogr.build_geogr_sid import (
    add_collision_leaf,
    build_geo_constrained_pairs,
    load_catalog,
    residual_kmeans,
)


class GeoGrSidBuilderTest(unittest.TestCase):
    def test_geo_constrained_pairs_require_behavior_and_distance(self):
        user_items = {
            1: {10, 20, 30},
            2: {10, 20},
            3: {10, 30},
        }
        coordinates = {
            10: (40.7500, -73.9900),
            20: (40.7510, -73.9910),
            30: (41.1000, -73.9900),
        }

        pairs, report = build_geo_constrained_pairs(
            user_items,
            coordinates,
            max_distance_km=3.0,
            min_common_users=2,
            swing_alpha=1.0,
        )

        self.assertEqual([(pair.left, pair.right) for pair in pairs], [(10, 20)])
        self.assertEqual(report["behavior_candidate_pairs"], 2)
        self.assertEqual(report["geo_retained_pairs"], 1)

    def test_residual_kmeans_returns_three_layers(self):
        embeddings = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [5.0, 5.0],
                [5.1, 5.0],
            ],
            dtype=np.float32,
        )

        codes, quantized, report = residual_kmeans(
            embeddings,
            codebook_size=2,
            num_layers=3,
            seed=7,
        )

        self.assertEqual(codes.shape, (4, 3))
        self.assertEqual(quantized.shape, embeddings.shape)
        self.assertEqual(len(report["layers"]), 3)

    def test_collision_leaf_makes_complete_paths_unique(self):
        raw = {
            3: [1, 2, 3],
            1: [1, 2, 3],
            2: [4, 5, 6],
        }

        final = add_collision_leaf(raw)

        self.assertEqual(final[1], [1, 2, 3, 0])
        self.assertEqual(final[3], [1, 2, 3, 1])
        self.assertEqual(final[2], [4, 5, 6])
        self.assertEqual(len({tuple(code) for code in final.values()}), 3)

    def test_loads_legacy_poi_columns_and_role_category_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poi_path = root / "poi_info.csv"
            role_path = root / "role_priors.csv"
            pd.DataFrame(
                {
                    "PoiId": [0],
                    "Latitude": [40.75],
                    "Longitude": [-73.99],
                }
            ).to_csv(poi_path, index=False)
            pd.DataFrame(
                {
                    "pid": [0],
                    "l1_label": [3],
                    "l2_label": [19],
                    "category_l1": ["Nightlife Spot"],
                    "category_l2": ["Bar"],
                }
            ).to_csv(role_path, index=False)

            catalog = load_catalog(poi_path, role_path)

        self.assertEqual(catalog[0].pid, 0)
        self.assertEqual(catalog[0].category_l1, "Nightlife Spot")
        self.assertEqual(catalog[0].category_l2, "Bar")

    def test_load_catalog_ignores_duplicate_coordinate_columns_in_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poi_path = root / "poi_info.csv"
            role_path = root / "role_priors.csv"
            pd.DataFrame(
                {
                    "PoiId": [0],
                    "Latitude": [40.75],
                    "Longitude": [-73.99],
                }
            ).to_csv(poi_path, index=False)
            pd.DataFrame(
                {
                    "pid": [0],
                    "l1_label": [3],
                    "l2_label": [19],
                    "category_l1": ["Nightlife Spot"],
                    "category_l2": ["Bar"],
                    "latitude": [0.0],
                    "longitude": [0.0],
                }
            ).to_csv(role_path, index=False)

            catalog = load_catalog(poi_path, role_path)

        self.assertEqual(catalog[0].latitude, 40.75)
        self.assertEqual(catalog[0].longitude, -73.99)


if __name__ == "__main__":
    unittest.main()
