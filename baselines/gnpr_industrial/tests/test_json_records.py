from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gnpr_baseline.json_records import load_json_records


class JsonRecordsTest(unittest.TestCase):
    def test_reads_json_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.json"
            path.write_text(json.dumps([{"x": 1}, {"x": 2}]), encoding="utf-8")
            self.assertEqual(load_json_records(path), [{"x": 1}, {"x": 2}])

    def test_reads_spark_json_directory_in_filename_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_SUCCESS").write_text("", encoding="utf-8")
            (root / "part-00001.json").write_text('{"x":2}\n', encoding="utf-8")
            (root / "part-00000.json").write_text('{"x":1}\n', encoding="utf-8")
            self.assertEqual(load_json_records(root), [{"x": 1}, {"x": 2}])

if __name__ == "__main__":
    unittest.main()
