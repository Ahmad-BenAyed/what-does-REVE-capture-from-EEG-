from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from task_2.classification.compare_mlp_cnn import run
from task_2.classification.visualiser_leace import METRICS, main as visualize
from task_2.classification.leace_mlp import FAMILIES
from task_2.paths import PROJECT_ROOT


class ModelComparison(unittest.TestCase):
    def fixture(self, root, kind):
        train, leace = root / kind, root / (kind + "_leace")
        train.mkdir()
        leace.mkdir()
        subjects = [f"s{i}" for i in range(10)]
        rows = pd.DataFrame(dict(row_index=np.arange(10), Subject=subjects, True_Label=np.arange(10) % 2,
                                 Predicted_Label=0, Probability_Class_0=.75, Probability_Class_1=.25))
        scores = pd.DataFrame({"Subject": subjects, **{key: .5 for key in METRICS}})
        features = [f"{family}{i:03}" for family, n in FAMILIES.items() for i in range(n)]
        source = dict(member_crc=4, member_size=10)
        config = dict(model_type=kind, completed=True, included_subjects=subjects,
                      classes=[0, 1], class_names=["a", "b"], metadata_sha256="meta", labels_sha256="labels",
                      input_shape=[10, 2, 3, 4], n_rows_used=10, source_embeddings=source)
        (train / "config.json").write_text(json.dumps(config))
        rows.to_csv(train / "predictions.csv", index=False)
        scores.to_csv(train / "fold_metrics.csv", index=False)
        (leace / "config.json").write_text(json.dumps(dict(model_type=kind, completed=True, controls=1, seed=42,
                                                          features=features, embeddings=source,
                                                          metadata_sha256="meta", labels_sha256="labels")))
        records = []
        for i, subject in enumerate(subjects):
            folder = leace / subject
            folder.mkdir()
            np.savez(folder / "fold.npz", test_rows=[i], train_rows=np.delete(np.arange(10), i),
                     classes=[0, 1], labels=[i % 2], original_probabilities=[[.75, .25]])
            for feature in features:
                for control in (0, 1):
                    record = dict(Feature=feature, Subject=subject, Control=control, Rank=1,
                                  Train_cov_residual=0., Probe_skill_original=.7, Probe_skill_erased=0.,
                                  Test_RMS_change=.3, Test_token_RMS_change=.4)
                    for metric in METRICS:
                        change = -.2 if kind == "cnn" and control == 0 else -.1
                        record.update({metric: .5 + change, "Original_" + metric: .5, "Delta_" + metric: change})
                    records.append(record)
        pd.DataFrame(records).to_csv(leace / "by_subject.csv", index=False)
        return train, leace

    def test_paired_full_report_and_output_guard(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mlp, mlp_leace = self.fixture(root, "mlp")
            cnn, cnn_leace = self.fixture(root, "cnn")
            args = SimpleNamespace(mlp=mlp, cnn=cnn, mlp_leace=mlp_leace, cnn_leace=cnn_leace,
                                   out=root / "report", classification_only=False)
            run(args)
            per_subject = pd.read_csv(args.out / "leace_par_sujet_et_caracteristique.csv")
            self.assertEqual(len(per_subject), 630)
            np.testing.assert_allclose(per_subject.CNN_minus_MLP_Gap_Balanced_Accuracy, 10)
            family = pd.read_csv(args.out / "leace_par_famille.csv")
            self.assertTrue((family.N_Subjects == 10).all())
            self.assertTrue((args.out / "heatmap_familles.png").is_file())
            self.assertTrue((args.out / "matrices_confusion.csv").is_file())
            figures = root / "figures"
            with patch.object(sys, "argv", ["visualiser_leace", "--results", str(cnn_leace), "--out", str(figures)]):
                visualize()
            self.assertIn("sur le CNN", (figures / "index.html").read_text(encoding="utf-8"))
            self.assertTrue((figures / "variabilite_familles.png").is_file())
            with self.assertRaises(FileExistsError):
                run(args)
            args.out = root / "unpaired"
            rows = pd.read_csv(cnn / "predictions.csv")
            rows.loc[[0, 1], "Subject"] = ["s1", "s0"]
            rows.to_csv(cnn / "predictions.csv", index=False)
            with self.assertRaisesRegex(ValueError, "different rows"):
                run(args)
            self.assertFalse(args.out.exists())

    def test_direct_runner_help_check_and_plan_are_read_only(self):
        script = PROJECT_ROOT / "task_2/run_task_2.py"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [
                (["--help"], 0, b"leace_cnn"),
                (["--step", "train_cnn", "--all-subjects", "--dry-run"], 0, b"results_cnn_tous_sujets"),
                (["--step", "train_cnn", "--dry-run"], 0, b"results_cnn_sans_P02"),
                (["--step", "leace_cnn", "--dry-run", "--out", "new"], 0, b"classification.leace_cnn"),
                (["--step", "visualize_cnn", "--dry-run", "--out", "new"], 0, b"--model cnn"),
                (["--step", "compare_models", "--dry-run", "--out", "new"], 0, b"classification.compare_mlp_cnn"),
                (["--step", "compare_models", "--check", "--mlp", "missing", "--out", "new"], 1, b"[ERROR]"),
            ]
            for cli, status, token in pairs:
                completed = subprocess.run([sys.executable, str(script), *cli], cwd=root,
                                           capture_output=True, timeout=20)
                self.assertEqual(completed.returncode, status, completed.stderr)
                self.assertIn(token, completed.stdout)
            self.assertEqual(list(root.iterdir()), [])
