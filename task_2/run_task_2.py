"""Run one Task 2 stage using its classification package."""
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
    from .paths import Paths, PROJECT_ROOT, TASK_ROOT
else:
    from check_inputs import check
    from paths import Paths, PROJECT_ROOT, TASK_ROOT


STEPS = {
    "train_mlp": "train_mlp_loso",
    "train_cnn": "train_cnn_loso",
    "evaluate_mlp": "evaluate_frozen_mlp",
    "evaluate_cnn": "evaluate_frozen_cnn",
    "leace": "leace_mlp",
    "leace_cnn": "leace_cnn",
    "visualize": "visualiser_leace",
    "visualize_cnn": "visualiser_leace",
    "compare_models": "compare_mlp_cnn",
}


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


def build_job(args, paths, extra):
    stage = args.step.split("_")[0]
    command = [sys.executable, "-m", f"task_2.classification.{STEPS[args.step]}"]
    for key in ("out", "data", "embeddings", "metadata", "segments", "checkpoint",
                "models", "family", "results", "reference", "mlp", "cnn", "mlp_leace", "cnn_leace"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    if stage == "compare":
        keys = {"mlp": "mlp_results_dir", "cnn": "cnn_all_results_dir",
                "mlp_leace": "leace_results_dir", "cnn_leace": "leace_cnn_results_dir",
                "out": "comparison_dir"}
        for name, key in keys.items():
            setattr(args, name, getattr(args, name) or paths.location(key))
        command += options(**{name: getattr(args, name) for name in keys})
        if args.classification_only:
            command.append("--classification-only")
        return stage, (args.step, command, extra, args.out)

    if stage == "visualize":
        cnn = args.step == "visualize_cnn"
        explicit_results = args.results
        args.results = explicit_results or paths.location("leace_cnn_results_dir" if cnn else "leace_results_dir")
        args.out = args.out or (args.results / "visualisations" if explicit_results
                                else paths.location("cnn_visualisations_dir" if cnn else "visualisations_dir"))
        command += options(results=args.results, out=args.out, model="cnn" if cnn else None)
        return stage, (args.step, command, extra, args.out)

    args.embeddings = args.embeddings or paths.input("embeddings", args.data)
    args.metadata = args.metadata or paths.input("metadata", args.data)
    inputs = options(embeddings=args.embeddings, metadata=args.metadata)
    if stage == "train":
        args.segments = args.segments or paths.input("segments", args.data)
        result_key = "cnn_all_results_dir" if args.all_subjects else args.step.removeprefix("train_") + "_results_dir"
        args.out = args.out or paths.location(result_key)
        if args.all_subjects:
            command.append("--all-subjects")
        command += options(segments=args.segments, out=args.out, embedding_key=args.embedding_key,
                           label_key=args.label_key, subject_column=args.subject_column)
        if args.resume:
            command.append("--resume")
    elif stage == "evaluate":
        if args.checkpoint is None or args.out is None:
            raise ValueError("Evaluation requires --checkpoint and --out.")
        command += options(checkpoint=args.checkpoint, reference=args.reference, out=args.out)
    else:
        args.family = args.family or paths.location("family_dir")
        cnn = args.step == "leace_cnn"
        args.models = args.models or paths.location("cnn_all_results_dir" if cnn else "mlp_results_dir")
        args.out = args.out or paths.location("leace_cnn_results_dir" if cnn else "leace_results_dir")
        command += options(family=args.family, models=args.models, out=args.out,
                           config=args.config.resolve() if args.config else None)
    return stage, (args.step, command + inputs, extra, args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     epilog="Pass experiment options after --.")
    parser.add_argument("--step", choices=STEPS, required=True)
    parser.add_argument("--deep", action="store_true", help="Check label values and provenance with --check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    stage_options = {
        "data": ("train", "evaluate", "leace"),
        "segments": ("train",),
        "embeddings": ("train", "evaluate", "leace"),
        "metadata": ("train", "evaluate", "leace"),
        "checkpoint": ("evaluate",),
        "reference": ("evaluate",),
        "models": ("leace",),
        "family": ("leace",),
        "results": ("visualize",),
        "embedding-key": ("train",),
        "label-key": ("train",),
        "subject-column": ("train",),
        "resume": ("train",),
        "all-subjects": ("train",),
        "classification-only": ("compare",),
        "mlp": ("compare",),
        "cnn": ("compare",),
        "mlp-leace": ("compare",),
        "cnn-leace": ("compare",),
    }
    for option in ("config", "out", *stage_options):
        if option in ("resume", "all-subjects", "classification-only"):
            parser.add_argument("--" + option, action="store_true")
        else:
            parser.add_argument("--" + option, type=str if option in
                                ("embedding-key", "label-key", "subject-column") else Path)
    raw, extra = split_args(list(sys.argv[1:] if argv is None else argv))
    args = parser.parse_args(raw)
    stage = args.step.split("_")[0]
    for option, stages in stage_options.items():
        if getattr(args, option.replace("-", "_")) and stage not in stages:
            parser.error(f"--{option} does not apply to {args.step}.")
    if args.all_subjects and args.step != "train_cnn":
        parser.error("--all-subjects applies to train_cnn.")
    if args.deep and (not args.check or stage not in ("train", "leace")):
        parser.error("--deep requires --check and a training or LEACE step.")
    managed = {"--" + key for key in ("out", "config", *stage_options)} | {"--help", "-h", "--model"}
    if extra and (args.check or any(managed_option(token, managed) for token in extra)):
        parser.error("Pass experiment options after --. Set infrastructure options before --.")
    args.embedding_key = args.embedding_key or "embeddings"
    args.label_key = args.label_key or "y"
    args.subject_column = args.subject_column or "subject"
    try:
        paths = Paths(args.config)
        stage, job = build_job(args, paths, extra)
        if args.dry_run:
            name, command, tail, out = job
            sensitive = any(token.split("=", maxsplit=1)[0].startswith(
                ("--test-sub", "--exclude-sub")) for token in tail)
            shown = command if sensitive else [*command, *tail]
            print(f"{name} | output={out} | {display(shown)}"
                  + (" [extra arguments omitted from display]" if sensitive else ""))
            return 0

        def validate():
            return check(args, paths, stage)

        if args.check:
            return 0 if validate() else 1
        return execute([job], PROJECT_ROOT, validate)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
