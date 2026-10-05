import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "geogr_full_pipeline"


class IndependentFullPipelineTest(unittest.TestCase):
    def test_full_pipeline_does_not_import_identifier_only_baseline(self) -> None:
        offenders = []
        for path in sorted(PACKAGE_ROOT.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            if "baselines.geogr" in text:
                offenders.append(path.name)
        self.assertEqual([], offenders)

    def test_internal_sid_utilities_exist(self) -> None:
        path = PACKAGE_ROOT / "sid_utils.py"
        text = path.read_text(encoding="utf-8")
        for name in (
            "CatalogPoi",
            "GeoPair",
            "build_geo_constrained_pairs",
            "load_catalog",
            "load_train_user_items",
            "residual_kmeans",
            "sid_tokens",
        ):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
