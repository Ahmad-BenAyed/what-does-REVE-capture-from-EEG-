import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from task_1.paths import PROJECT_ROOT, TASK_ROOT, Paths, family_script

EXPECTED = {"T": "results_temporal", "F": "results_spectral", "TF": "results_tf",
            "C": "results_c", "R": "results_r", "X": "results_x"}


class PathsTest(unittest.TestCase):
    def test_defaults_from_other_working_directory(self):
        before = Path.cwd()
        with TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                paths = Paths()
                for family, folder in EXPECTED.items():
                    with self.subTest(family=family):
                        self.assertEqual(paths.input("segments"), PROJECT_ROOT / "data/segments_labeled.npz")
                        self.assertEqual(paths.input("embeddings"), PROJECT_ROOT / "data/reve_embeddings.npz")
                        self.assertEqual(paths.family_result(family), TASK_ROOT / "family" / family / "resultats" / folder)
                        self.assertTrue(family_script(family).is_file())
                self.assertEqual(paths.location("visualisations_dir"), TASK_ROOT / "visualisations")
            finally:
                os.chdir(before)

    def test_config_relative_paths_and_architecture_fallback(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom.json"
            config.write_text(json.dumps({"data_dir": "incoming", "families": {"F": "new"}}))
            paths = Paths(config)
            self.assertEqual(paths.input("embeddings"), root / "incoming/reve_embeddings.npz")
            self.assertEqual(paths.location("family_dir"), TASK_ROOT / "family")
            self.assertEqual(paths.families["F"], "new")
            self.assertEqual(paths.families["T"], "results_temporal")

    def test_config_file(self):
        data = json.loads((TASK_ROOT / "config.json").read_text())
        self.assertEqual(data["families"], EXPECTED)

    def test_missing_default_config_falls_back(self):
        with TemporaryDirectory() as temporary:
            with patch("task_1.paths.TASK_ROOT", Path(temporary)):
                self.assertEqual(Paths().input("segments"),
                                 (Path(temporary) / "../data/segments_labeled.npz").resolve())
