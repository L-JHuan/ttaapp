import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_matched_pipeline.sh"
EVALUATE_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "evaluate.sh"


class RunMatchedPipelineScriptTest(unittest.TestCase):
    def test_full_pipeline_has_no_identifier_only_runtime_dependency(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("$REPO_ROOT/baselines/geogr:", text)

    def test_evaluation_uses_pipeline_level_test_dataset(self):
        text = SCRIPT.read_text(encoding="utf-8")
        block = re.search(
            r"evaluate_variant\(\) \{(?P<body>.*?)\n\}",
            text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(block)
        body = block.group("body")

        self.assertIn(
            'TEST_DATASET="$PIPELINE_RUN_ROOT/data/llm_test.json"',
            body,
        )
        self.assertNotIn(
            'TEST_DATASET="$RUN_ROOT/data/llm_test.json"',
            body,
        )
        self.assertIn('PYTHON_BIN="$PYTHON_BIN"', body)
        self.assertIn("${EVAL_RUN_SUFFIX}", text)

    def test_evaluation_uses_configured_python_interpreter(self):
        text = EVALUATE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('PYTHON_BIN=${PYTHON_BIN:-python}', text)
        self.assertEqual(text.count('"$PYTHON_BIN" -m'), 5)
        self.assertNotRegex(text, r"(?m)^\s*python -m ")


if __name__ == "__main__":
    unittest.main()
