"""Run Task 1 family scripts sequentially or generate figures."""
import argparse
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO
from datetime import datetime, timezone
import logging
import shlex
import subprocess
import sys
from time import monotonic
from uuid import uuid4

if __package__:
    from .check_inputs import check
    from .paths import Paths, SCRIPTS, TASK_ROOT, family_script
else:
    from check_inputs import check
    from paths import Paths, SCRIPTS, TASK_ROOT, family_script


def display(command):
    return subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)


def execute(jobs, cwd, validate):
    log_dir = TASK_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{timestamp}-{uuid4().hex}.log"
    logger = logging.Logger(TASK_ROOT.name)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path, mode="x", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    status = 1
    start = monotonic()
    try:
        logger.info("Start | log=%s", log_path)
        diagnostics = StringIO()
        with redirect_stdout(diagnostics):
            valid = validate()
        for line in diagnostics.getvalue().splitlines():
            label, _, message = line.partition(" ")
            level = {"[ERROR]": logging.ERROR, "[WARNING]": logging.WARNING}.get(label, logging.INFO)
            logger.log(level, "%s", message or label)
        if not valid:
            logger.warning("Preflight failed.")
            return status
        for name, command, extra, out in jobs:
            step_start = monotonic()
            logger.info("%s start | output=%s | %s%s", name, out, display(command),
                        " [extra arguments omitted from log]" if extra else "")
            result = subprocess.run([*command, *extra], cwd=cwd, check=False)
            status = result.returncode
            logger.log(logging.ERROR if status else logging.INFO,
                       "%s end | return=%d | %.2fs", name, status, monotonic() - step_start)
            if status:
                return status
        return status
    except KeyboardInterrupt:
        status = 130
        logger.warning("Interrupted")
        return status
    except OSError as error:
        status = 1
        logger.error("%s", error)
        return status
    finally:
        logger.info("End | status=%d | %.2fs", status, monotonic() - start)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)


def managed_option(token, managed):
    option = token.split("=", maxsplit=1)[0]
    return option in managed or (option.startswith("--") and
                                 any(key.startswith(option) for key in managed))


def split_args(argv):
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def options(**values):
    command = []
    for key, value in values.items():
        if value is not None:
            command.extend(("--" + key.replace("_", "-"), str(value)))
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     epilog="Pass family-specific options after --.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--family", nargs="+", choices=[*SCRIPTS, "all"])
    action.add_argument("--visualize", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--segments", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--out", type=Path, help="Single output folder or family tree for multiple families")
    parser.add_argument("--root", type=Path, help="Input family tree for visualization")
    parser.add_argument("--r-mode", choices=("global", "summary"), default="global")
    raw, extra = split_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(raw)
    if not (args.family or args.visualize or args.check):
        parser.error("Choose --family, --visualize or --check.")
    if args.check and args.dry_run:
        parser.error("Choose either --check or --dry-run.")
    if extra and (not args.family or args.check or
                  any(managed_option(option, ("--config", "--segments", "--embeddings", "--out", "--help", "-h"))
                      for option in extra)):
        parser.error("Pass family-specific options after --. Set infrastructure options before --.")
    try:
        paths = Paths(args.config)
        jobs = []
        root = None
        if args.visualize:
            root = args.root.resolve() if args.root else paths.location("family_dir")
            out = args.out.resolve() if args.out else paths.location("visualisations_dir")
            command = [sys.executable, str(TASK_ROOT / "visualiser_resultats.py"),
                       *options(config=args.config.resolve() if args.config else None,
                                root=root, out=out, r_mode=args.r_mode)]
            jobs.append(("visualize", command, [], out))
        else:
            selected = args.family or ["all"]
            families = list(SCRIPTS) if "all" in selected else list(dict.fromkeys(selected))
            segments = args.segments.resolve() if args.segments else paths.input("segments")
            embeddings = args.embeddings.resolve() if args.embeddings else paths.input("embeddings")
            for family in families:
                out = paths.family_result(family, args.out.resolve() if args.out else None)
                if args.out and len(families) == 1:
                    out = args.out.resolve()
                command = [sys.executable, str(family_script(family)),
                           *options(config=args.config.resolve() if args.config else None,
                                    segments=segments, embeddings=embeddings, out=out)]
                jobs.append((family, command, extra, out))

        def validate():
            return check(paths, [(name, out) for name, _, _, out in jobs],
                         segments=args.segments, embeddings=args.embeddings,
                         root=root, r_mode=args.r_mode)

        if args.dry_run:
            for name, command, tail, out in jobs:
                print(f"{name} | output={out} | {display([*command, *tail])}")
            return 0
        if args.check:
            return 0 if validate() else 1
        return execute(jobs, Path.cwd(), validate)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
