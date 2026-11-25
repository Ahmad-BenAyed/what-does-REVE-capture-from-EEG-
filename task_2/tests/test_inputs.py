from contextlib import redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from task_2 import run_task_2 as runner
from task_2.paths import FOLDS, Paths, output_problem


class InputsTest(unittest.TestCase):
    def test_leace_reports_per_fold_and_targets_with_no_segments_dependency(self):
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            status = runner.main(["--step", "leace", "--check", "--data", str(root),
                                  "--models", str(root / "models"), "--family", str(root / "family"),
                                  "--out", str(root / "new")])
            message = output.getvalue()
            self.assertEqual(status, 1)
            for fold in FOLDS:
                self.assertIn(str(Path(fold) / "checkpoint.pt"), message)
                self.assertIn(str(Path(fold) / "predictions.csv"), message)
            self.assertEqual(message.count("test_predictions.npz"), 6)
            self.assertNotIn("segments_labeled.npz", message)
            self.assertFalse((root / "new").exists())

    def test_visualize_checks_csv_without_arrays_or_checkpoints(self):
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            (root / "by_subject.csv").write_text("Feature,Subject,Control\nT001,example,0\n")
            with patch("task_2.check_inputs.check_dependencies", return_value=True):
                status = runner.main(["--step", "visualize", "--check", "--results", str(root)])
            self.assertEqual(status, 0)
            self.assertNotIn("checkpoint.pt", output.getvalue())
            self.assertNotIn("reve_embeddings.npz", output.getvalue())
            self.assertFalse((root / "visualisations").exists())

    def test_training_preflight_reads_npz_headers_and_metadata(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy unavailable")
        import numpy as np

        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            labels = np.array([1, 2])
            np.savez_compressed(root / "segments_labeled.npz", y=labels)
            np.savez_compressed(root / "reve_embeddings.npz", embeddings=np.zeros((2, 3, 4)), y=labels)
            metadata = root / "segments_metadata.csv"
            metadata.write_text("subject\na\nb\n")
            cli = ["--step", "train_mlp", "--check", "--data", str(root), "--out", str(root / "new")]
            with patch("task_2.check_inputs.check_dependencies", return_value=True):
                with patch("numpy.load", side_effect=AssertionError("preflight inflated an array")):
                    self.assertEqual(runner.main(cli), 0)
                    metadata.write_text("subject\na\n")
                    self.assertEqual(runner.main(cli), 1)
            self.assertIn("inconsistent row counts", output.getvalue())

    def test_frozen_preflight_preserves_checkpoint_defined_key_names(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy unavailable")
        import numpy as np

        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            embeddings = root / "embeddings.npz"
            metadata = root / "metadata.csv"
            with ZipFile(checkpoint, "w") as archive:
                archive.writestr("checkpoint/data.pkl", b"placeholder")
            metadata.write_text("custom_subject\nexample\n")
            cli = ["--step", "evaluate_mlp", "--check", "--checkpoint", str(checkpoint),
                   "--embeddings", str(embeddings), "--metadata", str(metadata),
                   "--out", str(root / "new")]
            with patch("task_2.check_inputs.check_dependencies", return_value=True):
                with ZipFile(embeddings, "w") as archive:
                    archive.writestr("custom_key.npy", b"placeholder")
                self.assertEqual(runner.main(cli), 1)
                np.savez(embeddings, custom_key=np.ones((2, 4)))
                self.assertEqual(runner.main(cli), 1)
                np.savez(embeddings, custom_key=np.ones((1, 4)))
                self.assertEqual(runner.main(cli), 0)

    def test_nonempty_output_and_resume_gate(self):
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            (root / "by_subject.csv").write_text("Feature,Subject,Control\nT001,example,0\n")
            (root / "config.json").write_text("{}")
            with patch("task_2.check_inputs.check_dependencies", return_value=True):
                status = runner.main(["--step", "visualize", "--check", "--results", str(root), "--out", str(root)])
            self.assertEqual(status, 1)
            self.assertIsNotNone(output_problem(root))
            self.assertIsNone(output_problem(root, resume=True))
            self.assertEqual((root / "config.json").read_text(), "{}")


    def test_leace_validates_all_target_and_checkpoint_headers(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy unavailable")
        import numpy as np

        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            root = Path(temporary)
            family_root = root / "family"
            models = root / "models"
            embeddings = root / "reve_embeddings.npz"
            np.savez_compressed(embeddings, embeddings=np.zeros((2, 4)), y=[0, 1])
            (root / "segments_metadata.csv").write_text("subject\na\nb\n")
            with ZipFile(embeddings) as archive:
                crc = archive.getinfo("embeddings.npy").CRC
            for family in Paths().families:
                folder = Paths().family_result(family, family_root)
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "test_predictions.npz", row_index=[0, 1],
                                    subject=["a", "b"], column_names=["feature"], y_true=[[0], [1]])
            for fold in FOLDS:
                folder = models / fold
                folder.mkdir(parents=True)
                with ZipFile(folder / "checkpoint.pt", "w") as archive:
                    archive.writestr("checkpoint/data.pkl", b"placeholder")
                (folder / "predictions.csv").write_text("row_index,Probability_Class_0\n0,1\n")
            config = models / "config.json"
            settings = {"input_shape": [2, 1, 1, 4], "source_embeddings": {"member_crc": crc},
                        "metadata_sha256": "unused", "labels_sha256": "unused"}
            config.write_text(json.dumps(settings))
            cli = ["--step", "leace", "--check", "--data", str(root), "--family", str(family_root),
                   "--models", str(models), "--out", str(root / "out")]
            with patch("task_2.check_inputs.check_dependencies", return_value=True):
                with patch("numpy.load", side_effect=AssertionError("preflight inflated an array")):
                    self.assertEqual(runner.main(cli), 0)
                    settings["source_embeddings"]["member_crc"] ^= 1
                    config.write_text(json.dumps(settings))
                    self.assertEqual(runner.main(cli), 1)
