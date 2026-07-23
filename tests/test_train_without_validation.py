import tempfile
import unittest
from pathlib import Path

from tap_sid.train_tap_sid import validate_dataset_paths


class ValidateDatasetPathsTest(unittest.TestCase):
    def test_allows_training_without_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.json"
            train.write_text("[]", encoding="utf-8")

            validate_dataset_paths(train, None, False)

    def test_eval_during_train_requires_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            train = Path(directory) / "train.json"
            train.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requires --valid_dataset"):
                validate_dataset_paths(train, None, True)
