import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_industrial_pipeline.sh"
EXAMPLE_ENV = ROOT / "configs" / "industrial.example.env"
LOCAL_ENV = ROOT / "configs" / "local.env"


class IndustrialPipelineScriptTest(unittest.TestCase):
    def test_one_click_runner_reuses_tap_spark_outputs(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('REUSE_PREPROCESSED', script)
        self.assertIn('sequence_parquet', script)
        self.assertIn('metadata/catalog', script)
        self.assertIn('prepare_industrial_inputs_spark', script)
        self.assertIn('tap_sid/build_llm_data_spark.py', script)
        self.assertNotIn('prepare_data_spark.sh', script)

    def test_one_click_runner_supports_train_test_only_protocol(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('NO_VALIDATION', script)
        self.assertIn('llm_train.jsonl', script)
        self.assertIn('llm_test.jsonl', script)
        self.assertIn('test_metrics.json', script)
        self.assertIn('CPT_SFT_ROOT', script)
        self.assertNotIn('SFT_ONLY_ROOT', script)

    def test_example_env_matches_industrial_handoff(self) -> None:
        text = EXAMPLE_ENV.read_text(encoding="utf-8")
        self.assertIn('REUSE_PREPROCESSED=1', text)
        self.assertIn('NO_VALIDATION=1', text)
        self.assertIn('GPUS=0,1,2,3', text)
        self.assertIn('TEST_BEAMS=10', text)
        self.assertIn('EM_BEAMS=20', text)

    def test_example_and_local_are_complete_industrial_configs(self) -> None:
        required = (
            'PROCESSED_ROOT=',
            'RUN_ROOT=',
            'BASE_MODEL=',
            'REUSE_PREPROCESSED=1',
            'NO_VALIDATION=1',
            'DATASET=Industrial',
            'TEST_BEAMS=10',
        )
        for path in (EXAMPLE_ENV, LOCAL_ENV):
            text = path.read_text(encoding="utf-8")
            for item in required:
                self.assertIn(item, text, f"{path.name} lacks {item}")
            self.assertNotIn('DATASET=NYC', text)
            self.assertNotIn('DATASET=TKY', text)

    def test_runner_accepts_explicit_config_or_discovers_one(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if [[ $# -ge 1 ]]', script)
        self.assertIn('configs/local.env', script)
        self.assertIn('configs/industrial.example.env', script)


if __name__ == "__main__":
    unittest.main()
