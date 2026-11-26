import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from task_1.loso_normalisation_sujet import STATS, load_inputs, loso, main as experiment, parse, ridge_predict, r2_and_r
from task_1.paths import TASK_ROOT
from task_1.visualiser_resultats import load_results


class InputsTest(unittest.TestCase):
    def test_saved_run_integrity(self):
        source = TASK_ROOT / "results_loso_norm_sujet"
        if not source.is_dir():
            self.skipTest("Saved run is unavailable")
        config, scores, features, families, identity, validation = load_results(source)
        self.assertEqual((config["n_trials"], config["n_targets"], config["n_subjects"]), (2642, 308, 10))
        self.assertEqual((len(scores), len(features), len(families)), (55440, 189, 18))
        self.assertEqual(set(scores.version), set("ABC"))
        self.assertEqual(set(scores.control), set(range(6)))
        self.assertEqual(identity.loc["bruts", "balanced_accuracy"], 1)
        self.assertTrue((scores.r2 < -100).any())

    def test_check_reads_paths_without_creating_output(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / name for name in ("metadata", "features", "pooled")]
            for source in inputs:
                source.touch()
            out = root / "unused"
            args = ["--metadata", str(inputs[0]), "--features", str(inputs[1]),
                    "--pooled", str(inputs[2]), "--out", str(out), "--check"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(experiment(args), 0)
                inputs[0].unlink()
                self.assertEqual(experiment(args), 1)
            self.assertFalse(out.exists())

    def test_optional_targets_are_actual_channels(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, pooled, metadata = [root / name for name in ("f.npz", "x.npy", "m.csv")]
            channels = np.arange(12).reshape(4, 3, 1)
            np.savez(features, per_channel=channels, global_features=np.ones((4, 1)),
                     channel_feature_names=["T001"], global_feature_names=["R001"])
            np.save(pooled, np.ones((4, 2)))
            metadata.write_text("subject\na\na\nb\nb\n")
            args = parse(["--features", str(features), "--pooled", str(pooled),
                          "--metadata", str(metadata), "--out", str(root / "other")])
            args.out = None
            _, y, _, dims, _ = load_inputs(args)
            self.assertEqual(list(dims.dimension), [*STATS, "global"])
            args.par_canal = True
            _, y, _, dims, _ = load_inputs(args)
            self.assertEqual(list(dims.dimension), ["canal_00", "canal_01", "canal_02", "global"])
            np.testing.assert_array_equal(y[:, :3], channels[:, :, 0])

    def test_loso_exports_validation_at_selected_alpha(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(15, 3))
        y = x[:, :2] + rng.normal(size=(15, 2))
        groups = np.repeat(["a", "b", "c"], 5)
        alphas = [0.1, 10.]
        subject, _, _, selected, validation = loso(x, y, groups, alphas, {})[0]
        self.assertEqual(subject, "a")
        rows = []
        for train, held in [("b", "c"), ("c", "b")]:
            preds = ridge_predict(x[groups == train], y[groups == train], x[groups == held], alphas)
            rows.append([r2_and_r(y[groups == held], p)[0] for p in preds])
        expected = np.mean(rows, axis=0)
        np.testing.assert_allclose(validation, expected.max(axis=0))
        np.testing.assert_array_equal(selected, np.asarray(alphas)[expected.argmax(axis=0)])
