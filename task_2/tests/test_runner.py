from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from task_2 import run_task_2 as runner
from task_2.paths import PROJECT_ROOT, TASK_ROOT


class RunnerTest(unittest.TestCase):
    def run_dry(self, *args):
        output = StringIO()
        with patch.object(runner.subprocess, "run") as child, redirect_stdout(output):
            self.assertEqual(runner.main([*args, "--dry-run"]), 0)
        child.assert_not_called()
        return output.getvalue()

    def test_planned_commands_and_no_output_changes(self):
        with TemporaryDirectory() as temporary:
            target = Path(temporary) / "new"
            output = self.run_dry("--step", "leace", "--out", str(target))
            self.assertIn("-m task_2.classification.leace_mlp", output)
            self.assertIn(str(TASK_ROOT / "family"), output)
            self.assertIn("results_mlp_tous_sujets", output)
            self.assertNotIn("train_mlp_loso", output)
            self.assertFalse(target.exists())

    def test_explicit_cli_overrides_config(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "custom.json"
            config.write_text(json.dumps({"data_dir": "configured"}))
            explicit = root / "explicit.npz"
            output = self.run_dry("--step", "train_mlp", "--config", str(config),
                                  "--embeddings", str(explicit))
            self.assertIn(str(explicit), output)
            self.assertNotIn(str(root / "configured/reve_embeddings.npz"), output)
            self.assertIn(str(root / "configured/segments_labeled.npz"), output)

    def test_visualize_only(self):
        output = self.run_dry("--step", "visualize")
        self.assertIn("task_2.classification.visualiser_leace", output)
        self.assertIn(str(TASK_ROOT / "classification/results_leace_mlp/visualisations"), output)
        self.assertNotIn("--embeddings", output)

    def test_logs_are_distinct_and_stop_after_error(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = ["example", "--out", str(root / "output")]
            jobs = [("first", command, ["PRIVATE_VALUE"], root / "output"),
                    ("second", command, [], root / "output")]
            with patch.object(runner, "TASK_ROOT", root), redirect_stderr(StringIO()):
                with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=7)) as child:
                    self.assertEqual(runner.execute(jobs, PROJECT_ROOT, lambda: True), 7)
                    self.assertEqual(runner.execute(jobs, PROJECT_ROOT, lambda: True), 7)
                self.assertEqual(child.call_count, 2)
                self.assertEqual(child.call_args.args[0][-1], "PRIVATE_VALUE")
                self.assertEqual(child.call_args.kwargs["cwd"], PROJECT_ROOT)
            logs = list((root / "logs").glob("*.log"))
            self.assertEqual(len(logs), 2)
            for path in logs:
                content = path.read_text()
                self.assertNotIn("PRIVATE_VALUE", content)
                self.assertNotIn("second start", content)
                self.assertIn("status=7", content)

    def test_passthrough_cannot_override_preflight(self):
        with self.assertRaises(SystemExit) as raised, redirect_stderr(StringIO()):
            runner.main(["--step", "train_cnn", "--", "--ou", "other"])
        self.assertEqual(raised.exception.code, 2)
