from contextlib import redirect_stdout
import importlib
from importlib.util import find_spec
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from task_1.check_inputs import check
from task_1.paths import Paths, SCRIPTS, ridge_csv


class InputsTest(unittest.TestCase):
    def test_missing_data_is_reported(self):
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            config = root / "config.json"
            config.write_text(json.dumps({"data_dir": "absent"}))
            self.assertFalse(check(Paths(config), [("T", root / "out")]))
            self.assertIn("[ERROR] Data directory missing:", output.getvalue())
            self.assertIn("segments_labeled.npz", output.getvalue())
            self.assertFalse((root / "out").exists())

    def test_visualization_preflight_uses_only_results(self):
        if any(find_spec(name) is None for name in ("numpy", "pandas", "matplotlib")):
            self.skipTest("Task 1 visualization dependencies unavailable")
        with TemporaryDirectory() as temporary, redirect_stdout(StringIO()) as output:
            root = Path(temporary)
            paths = Paths()
            for family in SCRIPTS:
                folder = paths.family_result(family, root)
                folder.mkdir(parents=True)
                (folder / ridge_csv(family)).touch()
            self.assertTrue(check(paths, [("visualize", root / "out")], root=root))
            self.assertNotIn("segments_labeled.npz", output.getvalue())

    def test_each_direct_family_refuses_nonempty_output(self):
        if any(find_spec(name) is None for name in ("numpy", "pandas", "scipy", "sklearn")):
            self.skipTest("Task 1 regression dependencies unavailable")
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            preserved = root / "result.csv"
            preserved.write_bytes(b"historical")
            for family, filename in SCRIPTS.items():
                with self.subTest(family=family):
                    module = importlib.import_module(f"task_1.family.{family}.code.{Path(filename).stem}")
                    with patch.object(sys, "argv", [filename, "--out", str(root)]):
                        with self.assertRaisesRegex(FileExistsError, "already contains results"):
                            module.main()
            self.assertEqual(preserved.read_bytes(), b"historical")
            self.assertEqual(list(root.iterdir()), [preserved])
