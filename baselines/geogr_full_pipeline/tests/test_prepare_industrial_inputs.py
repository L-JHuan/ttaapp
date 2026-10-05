import tempfile
import unittest
from pathlib import Path

from geogr_full_pipeline.prepare_industrial_inputs_spark import resolve_shared_layout


class PrepareIndustrialInputsTest(unittest.TestCase):
    def test_requires_existing_tap_spark_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "metadata/catalog",
                "mappings/category_l1",
                "mappings/category_l2",
                "sequence_parquet/train",
            ):
                (root / relative).mkdir(parents=True)
            layout = resolve_shared_layout(root)
        self.assertEqual(layout.catalog, root / "metadata" / "catalog")
        self.assertEqual(layout.train_sequences, root / "sequence_parquet" / "train")

    def test_rejects_incomplete_shared_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                resolve_shared_layout(Path(directory))


if __name__ == "__main__":
    unittest.main()
