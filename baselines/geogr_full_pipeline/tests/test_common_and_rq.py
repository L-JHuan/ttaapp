import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from geogr_full_pipeline.build_initial_rq_sid import write_initial_codebook
from geogr_full_pipeline.common import (
    encode_geohash,
    load_embeddings,
    public_poi_description,
    save_embeddings,
)
from geogr_full_pipeline.sid_utils import CatalogPoi


class CommonAndRqTest(unittest.TestCase):
    def test_geohash_matches_known_new_york_prefix(self):
        self.assertEqual(encode_geohash(40.7484, -73.9857, 7), "dr5ru6j")

    def test_public_description_uses_only_available_fields(self):
        poi = CatalogPoi(
            pid=1,
            latitude=40.7484,
            longitude=-73.9857,
            category_l1="Outdoors & Recreation",
            category_l2="Scenic Lookout",
        )
        text = public_poi_description(poi)
        self.assertIn("geohash: dr5ru6j", text)
        self.assertIn("category level 2: Scenic Lookout", text)
        self.assertNotIn("brand", text.lower())

    def test_embedding_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.npz"
            values = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            save_embeddings(path, [3, 7], values)
            pids, loaded = load_embeddings(path)
        self.assertEqual(pids, [3, 7])
        np.testing.assert_allclose(loaded, values)

    def test_initial_codebook_keeps_three_tokens_without_leaf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sid.csv"
            codes = np.asarray([[1, 2, 3], [1, 2, 3]], dtype=np.int64)
            write_initial_codebook(path, [10, 20], codes)
            frame = pd.read_csv(path)
        self.assertEqual(frame["sid"].tolist(), ["[1, 2, 3]", "[1, 2, 3]"])
        self.assertEqual(frame["sid_tokens"].tolist(), ["<a_1><b_2><c_3>"] * 2)


if __name__ == "__main__":
    unittest.main()
