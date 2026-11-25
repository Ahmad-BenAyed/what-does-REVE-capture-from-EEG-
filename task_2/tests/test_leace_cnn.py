import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from task_2.classification.cnn_reve import EmbeddingCNN, mean_pool_embeddings, predict
from task_2.classification.evaluate_frozen_cnn import main as evaluate
from task_2.classification.leace_cnn import (
    erase_feature, fit_whitening, infer, random_factors, run_interventions,
    transform_tokens, validate_baselines,
)
from task_2.classification.leace_mlp import FAMILIES, read_targets
from task_2.paths import Paths, FOLDS


class CNNLEACE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def scenario(self):
        generator = np.random.default_rng(17)
        x = generator.normal(size=(72, 2, 3, 4)).astype(np.float32)
        x[..., 0] += generator.normal(size=(72, 1, 1)).astype(np.float32) * 3
        pooled = mean_pool_embeddings(x)[:, 0, 0].astype(float)
        values = pooled[:, :1].copy()
        model = EmbeddingCNN(x.shape[1:], 2, width=4).eval().requires_grad_(False)
        return x, pooled, values, model, np.arange(54), np.arange(54, 72)

    def test_affine_tokens_commute_with_mean_pooling(self):
        x, _, _, _, _, _ = self.scenario()
        rng = np.random.default_rng(1)
        mean, left, right = rng.normal(size=4), rng.normal(size=(4, 2)), rng.normal(size=(2, 4))
        transformed = transform_tokens(x, mean, left, right)
        self.assertEqual(transformed.shape, x.shape)
        np.testing.assert_allclose(transformed.mean((1, 2)),
                                   transform_tokens(x.astype(float).mean((1, 2)), mean, left, right),
                                   atol=1e-12)

    def test_integration_erases_pooled_signal_in_batches_and_preserves_model(self):
        x, pooled, values, model, train, test = self.scenario()
        weights = {name: weight.clone() for name, weight in model.state_dict().items()}
        labels = np.arange(len(x)) % 2
        probs, _ = infer(model, x, test, 5, "cpu")
        np.testing.assert_allclose(probs, predict(model, x, labels, test, 5, "cpu"), atol=1e-7)
        whitening = fit_whitening(pooled[train])
        args = (model, x, pooled, train, test, values, whitening)
        first = list(erase_feature(*args, np.random.default_rng(42), 1, 5, "cpu"))
        perturbed = values.copy()
        perturbed[test] *= -100
        second = list(erase_feature(*args[:5], perturbed, whitening, np.random.default_rng(42), 1, 7, "cpu"))
        self.assertGreater(first[0][3]["Probe_skill_original"], .8)
        self.assertLess(first[0][3]["Probe_skill_erased"], .1)
        self.assertLess(first[0][3]["Train_cov_residual"], 1e-7)
        for initial, altered in zip(first, second):
            np.testing.assert_array_equal(initial[1], altered[1])
            np.testing.assert_array_equal(initial[2], altered[2])
            np.testing.assert_allclose(initial[0], altered[0], atol=1e-7)
            self.assertEqual(initial[3]["Rank"], 1)
        _, _, _, s, vt = whitening
        left, right = random_factors(np.random.default_rng(42), s, vt, 1)
        np.testing.assert_array_equal(first[1][1], left)
        np.testing.assert_array_equal(first[1][2], right)
        for name, weight in model.state_dict().items():
            torch.testing.assert_close(weight, weights[name], rtol=0, atol=0)

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            settings = SimpleNamespace(seed=42, controls=1, batch_size=5, save_embeddings=True, out=out)
            folds = [(np.arange(len(x)), model, {"test_subject": "subject", "classes": [0, 1]},
                      train, test, labels[test], probs)]
            run_interventions(settings, folds, {"T001": values}, x, pooled, labels, "cpu")
            frame = pd.read_csv(out / "by_subject.csv")
            self.assertEqual(list(frame.Control), [0, 1])
            self.assertIn("Delta_Balanced_Accuracy", frame)
            self.assertIn("Test_token_RMS_change", frame)
            self.assertTrue((out / "summary.csv").is_file())
            with np.load(out / "subject/T001.npz") as saved:
                np.testing.assert_array_equal(saved["left"], first[0][1])
            self.assertEqual(np.load(out / "subject/T001_tokens.npy").shape, x[test].shape)

    def test_checkpoint_baselines_and_rejected_wrong_split(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            subjects = np.repeat([f"s{i}" for i in range(10)], 2)
            labels = np.tile([0, 1], 10)
            x = np.random.default_rng(0).normal(size=(20, 2, 3, 4)).astype("f4")
            model = EmbeddingCNN(x.shape[1:], 2, width=4).eval().requires_grad_(False)
            config = dict(included_subjects=np.unique(subjects).tolist(), completed=True, n_rows_used=20,
                          metadata_sha256="meta", classes=[0, 1], embedding_key="embeddings",
                          label_key="y", subject_column="subject")
            folds = []
            for name, subject in zip(FOLDS, config["included_subjects"]):
                test = np.flatnonzero(subjects == subject)
                checkpoint = dict(classes=[0, 1], test_subject=subject, train_rows=np.flatnonzero(subjects != subject),
                                  test_rows=test, model_config={}, test_labels=labels[test],
                                  metadata_sha256="meta", source_shape=list(x.shape), embedding_key="embeddings",
                                  label_key="y", subject_column="subject",
                                  train_subjects=[v for v in config["included_subjects"] if v != subject])
                path = p / name / "checkpoint.pt"
                path.parent.mkdir()
                pred, _ = infer(model, x, test, 2, "cpu")
                pd.DataFrame(dict(row_index=test, Subject=subject, True_Label=labels[test],
                                  Probability_Class_0=pred[:, 0], Probability_Class_1=pred[:, 1])).to_csv(
                                      path.parent / "predictions.csv", index=False)
                folds.append((path, model, checkpoint))
            self.assertEqual(len(list(validate_baselines(folds, x, labels, subjects, config, 1, "cpu"))), 10)
            folds[0][2]["train_rows"][0] = 0
            with self.assertRaisesRegex(ValueError, "LOSO split"):
                list(validate_baselines(folds, x, labels, subjects, config, 1, "cpu"))

    def test_frozen_reference_replays_a_real_small_cnn_checkpoint(self):
        x, _, _, model, train, test = self.scenario()
        labels = np.arange(len(x)) % 2
        subjects = np.where(np.arange(len(x)) < len(train), "train", "test")
        model.input_mean.fill_(.25)
        model.input_std.fill_(1.5)
        with TemporaryDirectory() as temp:
            p = Path(temp)
            metadata = p / "metadata.csv"
            pd.DataFrame({"subject": subjects}).to_csv(metadata, index=False)
            np.savez(p / "inputs.npz", embeddings=x, y=labels)
            torch.save(dict(format_version=1, model_state=model.state_dict(), classes=[0, 1],
                            source_shape=list(x.shape), embedding_key="embeddings", label_key="y",
                            model_config=dict(input_shape=list(x.shape[1:]), width=4, n_classes=2),
                            subject_column="subject", test_rows=test.tolist(), test_subject="test",
                            train_subjects=["train"], test_labels=labels[test].tolist(),
                            metadata_sha256=hashlib.sha256(metadata.read_bytes()).hexdigest()), p / "model.pt")
            baseline, _ = infer(model, x, test, 5, "cpu")
            reference = p / "reference.csv"
            pd.DataFrame(dict(row_index=test, Subject="test", True_Label=labels[test],
                              Probability_Class_0=baseline[:, 0], Probability_Class_1=baseline[:, 1])).to_csv(
                                  reference, index=False)
            cli = ["evaluate_frozen_cnn", "--checkpoint", str(p / "model.pt"),
                   "--embeddings", str(p / "inputs.npz"), "--metadata", str(metadata),
                   "--reference", str(reference), "--out", str(p / "results"), "--threads", "1", "--device", "cpu"]
            with patch.object(sys, "argv", cli):
                evaluate()
            result = json.loads((p / "results/metrics.json").read_text())
            self.assertTrue(result["Reference_probabilities_match"])
            self.assertLess(result["Max_absolute_probability_change"], 1e-6)

    def test_target_reader_aligns_subjects_and_catalogue(self):
        paths = Paths()
        subjects = np.array(["a", "b"])
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for family, count in FAMILIES.items():
                folder = paths.family_result(family, root)
                folder.mkdir(parents=True)
                np.savez(folder / "test_predictions.npz", row_index=[0, 1], subject=subjects,
                         column_names=[f"{family}{i:03}:ch0" for i in range(count)], y_true=np.zeros((2, count)))
            self.assertEqual(len(read_targets(root, subjects, np.arange(2))), 63)
            with self.assertRaisesRegex(ValueError, "alignement"):
                read_targets(root, subjects[::-1], np.arange(2))
