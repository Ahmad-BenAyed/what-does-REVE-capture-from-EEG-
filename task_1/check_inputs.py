"""Lightweight preflight for Task 1."""

if __package__:
    from .paths import SCRIPTS, family_script, output_problem, ridge_csv
else:
    from paths import SCRIPTS, family_script, output_problem, ridge_csv

import os
from importlib.util import find_spec


def check_output(path, resume=False):
    problem = output_problem(path, resume)
    if problem:
        print(f"[WARNING] {problem}")
        return False
    parent = next(folder for folder in (path, *path.parents) if folder.exists())
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        print(f"[ERROR] Output location is not writable: {path}")
        return False
    print(f"[OK] Output location: {path}")
    return True


def check_file(path):
    try:
        with path.open("rb"):
            pass
    except OSError as error:
        print(f"[ERROR] {path}: {error.strerror}")
        return False
    print(f"[OK] Readable file: {path}")
    return True


def check_dependencies(names):
    results = []
    for name in names:
        found = find_spec(name) is not None
        print(f"{'[OK]' if found else '[ERROR]'} Python dependency {name}: "
              f"{'available' if found else 'missing'}")
        results.append(found)
    return all(results)


def check(paths, jobs, *, segments=None, embeddings=None, root=None, r_mode="global"):
    visualise = root is not None
    results = [check_dependencies(("numpy", "pandas", "matplotlib") if visualise
                                  else ("numpy", "pandas", "scipy", "sklearn"))]
    if visualise:
        for family in SCRIPTS:
            folder = paths.family_result(family, root)
            if family == "R" and r_mode == "summary":
                folder = folder.with_name("results_r_summary")
            results.append(check_file(folder / ridge_csv(family)))
    else:
        data_dir = paths.location("data_dir")
        if data_dir.is_dir():
            print(f"[OK] Data directory: {data_dir}")
        elif segments is None or embeddings is None:
            print(f"[ERROR] Data directory missing: {data_dir}")
        else:
            print(f"[WARNING] Default data directory absent; using explicit inputs: {data_dir}")
        results.extend((check_file(segments or paths.input("segments")),
                        check_file(embeddings or paths.input("embeddings"))))
        results.extend(check_file(family_script(family)) for family, _ in jobs)
    results.extend(check_output(out) for _, out in jobs)
    return all(results)


def main():
    if __package__:
        from .run_task_1 import main as run_main
    else:
        from run_task_1 import main as run_main
    import sys

    return run_main(["--check", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
