import unittest
from unittest.mock import patch

from task_1.run_task_1 import main


class RunnerTest(unittest.TestCase):
    def test_no_command_is_safe(self):
        with patch("task_1.run_task_1.loso_normalisation_sujet.main") as run:
            self.assertEqual(main([]), 0)
            run.assert_not_called()

    def test_dispatch_preserves_arguments_and_status(self):
        for option, module in [("experiment", "loso_normalisation_sujet"), ("visualiser", "visualiser_resultats")]:
            with self.subTest(option=option), patch(f"task_1.run_task_1.{module}.main", return_value=7) as run:
                self.assertEqual(main([option, "--check"]), 7)
                run.assert_called_once_with(["--check"])
