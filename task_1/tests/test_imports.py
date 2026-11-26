import ast
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from task_1.paths import PROJECT_ROOT, TASK_ROOT


class ImportsTest(unittest.TestCase):
    def test_no_cross_task_or_legacy_imports(self):
        banned = {"task_2", "legacy", "03_extract_63_features", "04_regression_63"}
        for path in TASK_ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                else:
                    continue
                self.assertFalse({name.split(".")[0] for name in names} & banned, str(path))

    def test_imports_without_loading_arrays_or_checkpoints(self):
        code = """
import importlib
from importlib.util import find_spec
from pathlib import Path
import sys
from unittest.mock import patch
from contextlib import ExitStack

sys.path.insert(0, sys.argv[1])
required = ["numpy","pandas","sklearn","matplotlib"]
missing = [name for name in required if find_spec(name) is None]
if missing:
    print("Missing dependencies: " + ", ".join(missing))
    raise SystemExit(77)
import numpy

with ExitStack() as stack:
    stack.enter_context(patch("numpy.load", side_effect=AssertionError("array load on import")))

    root = Path(sys.argv[1])
    for path in sorted((root / "task_1").rglob("*.py")):
        if "tests" not in path.parts and path.name != "__init__.py":
            importlib.import_module(".".join(path.relative_to(root).with_suffix("").parts))
assert not Path("data").exists()
"""
        with TemporaryDirectory() as temporary:
            result = subprocess.run([sys.executable, "-c", code, str(PROJECT_ROOT)],
                                    cwd=temporary, capture_output=True, text=True, timeout=150)
            if result.returncode == 77:
                self.skipTest(result.stdout.strip())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_runner_help_from_an_empty_directory(self):
        with TemporaryDirectory() as temporary:
            for file in ("run_task_1.py", "check_inputs.py"):
                result = subprocess.run([sys.executable, str(TASK_ROOT / file), "--help"],
                                        cwd=temporary, capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)
