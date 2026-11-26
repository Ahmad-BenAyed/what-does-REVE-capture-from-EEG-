import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from task_1.paths import TASK_ROOT, Paths, output_problem


class PathsTest(unittest.TestCase):
    def test_relative_configuration_from_any_working_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom.json"
            config.write_text(json.dumps({"data_dir": "incoming", "results_dir": "scores"}))
            paths = Paths(config)
            self.assertEqual(paths.input("embeddings"), root / "incoming/reve_embeddings.npz")
            self.assertEqual(paths.input("features"), root / "incoming/features_63.npz")
            self.assertEqual(paths.input("metadata"), root / "incoming/segments_metadata.csv")
            self.assertEqual(paths.location("results_dir"), root / "scores")
            self.assertEqual(paths.location("visualisations_dir"), TASK_ROOT / "visualisations")

    def test_results_cannot_be_overwritten(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            self.assertIsNone(output_problem(out))
            (out / "scores_par_sujet.csv").touch()
            self.assertIn("already contains results", output_problem(out))
