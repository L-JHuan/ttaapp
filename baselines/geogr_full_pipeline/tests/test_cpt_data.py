import csv
import json
import tempfile
import unittest
from pathlib import Path

from geogr_full_pipeline.build_cpt_data import build_rows
from geogr_full_pipeline.train_cpt import CptDataset, make_collate


class TinyTokenizer:
    eos_token = "<eos>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens, truncation, max_length):
        del add_special_tokens, truncation
        return {"input_ids": list(range(1, min(len(text), max_length) + 1))}


class CptDataTest(unittest.TestCase):
    def test_cpt_dataset_uses_full_text_labels(self):
        tokenizer = TinyTokenizer()
        dataset = CptDataset([{"text": "abc"}], tokenizer, cutoff_len=8)
        batch = make_collate(tokenizer)([dataset[0]])
        self.assertEqual(batch["input_ids"].tolist(), batch["labels"].tolist())

    def test_builds_four_training_only_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "poi.csv").write_text(
                "pid,latitude,longitude\n1,40.0,-73.0\n2,40.1,-73.1\n",
                encoding="utf-8",
            )
            (root / "role.csv").write_text(
                "pid,l1_label,l2_label,category_l1,category_l2\n"
                "1,0,0,Food,Cafe\n2,1,1,Travel,Station\n",
                encoding="utf-8",
            )
            (root / "sid.csv").write_text(
                'pid,sid,sid_tokens\n1,"[0, 0, 0]",<a_0><b_0><c_0>\n'
                '2,"[0, 0, 1]",<a_0><b_0><c_1>\n',
                encoding="utf-8",
            )
            with (root / "train.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["sequence_PoiId", "sequence_UTCTimeOffset"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "sequence_PoiId": "[1, 2]",
                        "sequence_UTCTimeOffset": "['t1', 't2']",
                    }
                )
            rows, report = build_rows(
                root / "poi.csv",
                root / "role.csv",
                None,
                root / "train.csv",
                root / "sid.csv",
            )
            self.assertEqual(len(rows), 7)
            self.assertEqual(
                {row["template"] for row in rows},
                {
                    "poi_description_generation",
                    "poi_structured_information_alignment",
                    "poi_qa_definition",
                    "user_behavior_trajectory_modeling",
                },
            )
            self.assertEqual(report["train_sequence_rows"], 1)
            self.assertIn("<a_0><b_0><c_1>", json.dumps(rows))


if __name__ == "__main__":
    unittest.main()
