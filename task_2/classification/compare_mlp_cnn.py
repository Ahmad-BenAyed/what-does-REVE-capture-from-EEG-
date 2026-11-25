"""Compare existing MLP and CNN experiments with strict subject/row pairing."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import warnings

from ..paths import Paths, output_problem, prepare_output
from .visualiser_leace import FAMILIES, METRICS, load_results, table

def read_config(folder):
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))


def load_original(folder, kind):
    config = read_config(folder)
    if (config.get("model_type") != kind or not config.get("completed")
            or config.get("partial_run") or config.get("exclude_subjects")):
        raise ValueError(f"Completed all-subject {kind.upper()} results required: {folder}")
    rows = pd.read_csv(folder / "predictions.csv", dtype={"Subject": str, "True_Label": str, "Predicted_Label": str})
    scores = pd.read_csv(folder / "fold_metrics.csv", dtype={"Subject": str}).set_index("Subject").sort_index()
    subjects = set(config["included_subjects"])
    classes = list(map(str, config["classes"]))
    if (len(subjects) != 10 or set(rows.Subject) != subjects or set(scores.index) != subjects
            or rows.row_index.dtype.kind not in "iu" or not rows.row_index.is_unique
            or not scores.index.is_unique or len(rows) != config["n_rows_used"]
            or rows.row_index.min() < 0 or rows.row_index.max() >= config["input_shape"][0]
            or not set(rows.True_Label).issubset(classes) or not set(rows.Predicted_Label).issubset(classes)):
        raise ValueError(f"Inconsistent test rows, subjects or labels: {folder}")
    cols = [f"Probability_Class_{i}" for i in range(len(classes))]
    probabilities = rows[cols].to_numpy()
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1.)
            or not np.array_equal(np.asarray(classes)[probabilities.argmax(axis=1)], rows.Predicted_Label)
            or not np.isfinite(scores[list(METRICS)].to_numpy()).all()):
        raise ValueError(f"Inconsistent prediction probabilities or scores: {folder}")
    return config, rows.set_index("row_index").sort_index(), scores[list(METRICS)]


def compare_original(mlp, cnn):
    a, mrows, ms = load_original(mlp, "mlp")
    b, crows, cs = load_original(cnn, "cnn")
    keys = ("classes", "class_names", "metadata_sha256", "labels_sha256", "input_shape")
    if (any(a[key] != b[key] for key in keys)
            or a["source_embeddings"]["member_crc"] != b["source_embeddings"]["member_crc"]
            or a["source_embeddings"]["member_size"] != b["source_embeddings"]["member_size"]
            or not mrows[["Subject", "True_Label"]].equals(crows[["Subject", "True_Label"]])
            or not ms.index.equals(cs.index)):
        raise ValueError("MLP and CNN have different rows, cohorts, classes or source fingerprints. "
                         "Paired results would be misleading.")
    paired = ms.add_prefix("MLP_").join(cs.add_prefix("CNN_")).join((cs - ms).add_prefix("CNN_minus_MLP_"))
    return a, b, mrows, crows, paired


def load_erasure(folder, config, rows, kind):
    erasure = read_config(folder)
    source = config["source_embeddings"]
    if (erasure["metadata_sha256"] != config["metadata_sha256"]
            or erasure["embeddings"]["member_crc"] != source["member_crc"]
            or erasure["embeddings"]["member_size"] != source["member_size"]
            or (kind == "cnn" and (not erasure.get("completed") or erasure.get("model_type") != "cnn"
                                   or erasure.get("labels_sha256") != config["labels_sha256"]))):
        raise ValueError(f"LEACE provenance differs from classification: {folder}")
    data, controls = load_results(folder)
    if (set(data.index.get_level_values("Subject")) != set(rows.Subject)
            or set(data.index.get_level_values("Feature")) != set(erasure["features"])
            or controls != erasure["controls"]):
        raise ValueError(f"LEACE cohort, features or controls are inconsistent: {folder}")
    for subject, part in rows.groupby("Subject"):
        if Path(subject).name != subject or subject in (".", ".."):
            raise ValueError("Invalid subject directory name")
        with np.load(folder / subject / "fold.npz", allow_pickle=False) as fold:
            test = fold["test_rows"]
            if (len(test) != len(part) or not np.array_equal(np.sort(test), part.index)
                    or not np.array_equal(np.sort(fold["train_rows"]), rows.index[rows.Subject != subject])
                    or fold["classes"].tolist() != config["classes"]
                    or not np.array_equal(fold["labels"].astype(str), part.loc[test, "True_Label"])
                    or not np.allclose(fold["original_probabilities"],
                                       part.loc[test].filter(regex=r"^Probability_Class_").to_numpy(), atol=1e-5)):
                raise ValueError(f"LEACE test rows or baseline differ from classification: {folder / subject}")
    raw = pd.read_csv(folder / "by_subject.csv")
    controls_expected = set(range(erasure["controls"] + 1))
    if set(raw.Control) != controls_expected:
        raise ValueError(f"Unexpected LEACE control indices: {folder}")
    return erasure, data


def compare_erasure(mlp, cnn, a, b, mrows, crows):
    ma, m = load_erasure(mlp, a, mrows, "mlp")
    ca, c = load_erasure(cnn, b, crows, "cnn")
    if not m.index.equals(c.index) or (ma["controls"], ma["seed"], ma["features"]) != (ca["controls"], ca["seed"], ca["features"]):
        raise ValueError("LEACE subjects, features, control counts or seeds differ")
    effects = [prefix + metric for metric in METRICS for prefix in ("Loss_", "Random_loss_", "Gap_")]

    diagnostics = ["Rank", "Probe_skill_original", "Probe_skill_erased", "Train_cov_residual",
                   "Test_RMS_change", "Ratio_RMS"]
    detail = m[effects + diagnostics].add_prefix("MLP_").join(
        c[effects + diagnostics + ["Test_token_RMS_change"]].add_prefix("CNN_")).join(
        (c[effects] - m[effects]).add_prefix("CNN_minus_MLP_"))
    detail["Rank_differs"] = (m.Rank != c.Rank).astype(int)
    detail["Family"] = detail.index.get_level_values("Feature").str.extract(r"^([A-Z]+)\d", expand=False)
    if set(detail.Family) != set(FAMILIES):
        raise ValueError("Unexpected EEG families")
    return detail


def summary(frame, by):
    grouped = frame.groupby(by)
    stats = grouped.agg(["mean", "std"])
    stats.columns = ["_".join(col) for col in stats.columns]
    stats["N_Subjects"] = grouped.size()
    return stats


def plot_erasure(out, feature_summary, family_subject):
    columns = ["MLP_Gap_Balanced_Accuracy", "CNN_Gap_Balanced_Accuracy"]
    family = family_subject.groupby("Family")[columns].mean().reindex(FAMILIES)
    with PdfPages(out / "comparaison_leace.pdf") as pdf:
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        limit = max(1, np.abs(family.to_numpy()).max())
        image = ax.imshow(family, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        ax.set(xticks=[0, 1], xticklabels=["MLP", "CNN"], yticks=range(len(FAMILIES)),
               yticklabels=FAMILIES, title="Perte ciblée moins contrôle (points)")
        fig.colorbar(image, ax=ax)
        fig.savefig(out / "heatmap_familles.png", dpi=145)
        pdf.savefig(fig)
        plt.close(fig)
        for label in FAMILIES:
            sub = feature_summary.loc[feature_summary.Family == label]
            order = sub.sort_values("CNN_Gap_Balanced_Accuracy_mean")
            fig, ax = plt.subplots(figsize=(9, max(3, .37 * len(order))), layout="constrained")
            y = np.arange(len(order))
            for i, col in enumerate(columns):
                ax.errorbar(order[col + "_mean"], y + (i - .5) * .23,
                            xerr=order[col + "_std"], fmt="o", label=col.split("_")[0])
            ax.set(yticks=y, yticklabels=order.index, xlabel="Perte ciblée moins contrôle (points)",
                   title=f"{label} · moyenne et écart-type entre sujets")
            ax.axvline(0, color="gray", lw=1)
            ax.legend()
            fig.savefig(out / f"famille_{label}.png", dpi=145)
            pdf.savefig(fig)
            plt.close(fig)


def run(args):
    problem = output_problem(args.out)
    if problem:
        raise FileExistsError(problem)
    a, b, mrows, crows, original = compare_original(args.mlp, args.cnn)
    detail = None if args.classification_only else compare_erasure(
        args.mlp_leace, args.cnn_leace, a, b, mrows, crows)
    prepare_output(args.out)
    original.to_csv(args.out / "classification_par_sujet.csv")
    original_summary = original.agg(["mean", "std"]).T
    original_summary.to_csv(args.out / "classification_resume.csv")
    classes = list(map(str, a["classes"]))
    matrices = []
    for kind, rows in (("MLP", mrows), ("CNN", crows)):
        for subject, part in [("all", rows), *rows.groupby("Subject")]:
            matrix = pd.crosstab(part.True_Label, part.Predicted_Label)
            matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
            matrices.append(matrix.stack().rename("Count").reset_index().assign(Model=kind, Subject=subject))
    pd.concat(matrices, ignore_index=True).to_csv(args.out / "matrices_confusion.csv", index=False)
    page = ['<!doctype html><html lang="fr"><meta charset="utf-8"><title>MLP et CNN</title>',
            "<h1>Comparaison appariée MLP et CNN</h1>",
            "<p>Les pipelines diffèrent par le classifieur et par l’agrégation. Les écarts-types décrivent "
            "la variabilité entre sujets. Les essais et les caractéristiques ne sont pas des répétitions indépendantes.</p>",
            table(original.reset_index()), table(original_summary.reset_index(names="Indicateur"))]
    if detail is not None:
        if detail.Rank_differs.any():
            warnings.warn("Some paired interventions have different ranks. See Rank_differs in the subject CSV.")
        detail.to_csv(args.out / "leace_par_sujet_et_caracteristique.csv")
        by_feature = summary(detail.drop(columns="Family"), "Feature")
        by_feature["Family"] = detail.groupby("Feature").Family.first()
        for kind in ("MLP", "CNN"):
            for metric in METRICS:
                for prefix in ("Loss_", "Gap_"):
                    key = kind + "_" + prefix + metric
                    by_feature[key + "_rank"] = by_feature[key + "_mean"].rank(ascending=False, method="min")
        by_feature.to_csv(args.out / "leace_par_caracteristique.csv")
        effects = [name for name in detail if "_Loss_" in name or "_Random_loss_" in name or "_Gap_" in name]
        # Average features within subject before calculating subject variability.
        family_subject = detail.reset_index().groupby(["Family", "Subject"])[effects].mean()
        family_subject.to_csv(args.out / "leace_familles_par_sujet.csv")
        summary(family_subject, "Family").to_csv(args.out / "leace_par_famille.csv")
        plot_erasure(args.out, by_feature, family_subject)
        if detail.Rank_differs.any():
            page.append(f"<p>Rangs diff?rents pour {detail.Rank_differs.sum()} paires sujet-caract?ristique.</p>")
        page.extend(["<h2>LEACE</h2><p>Une perte positive indique une dégradation. L’effet relatif "
                     "soustrait la perte du contrôle aléatoire. Interpréter avec les diagnostics d’effacement.</p>",
                     '<img src="heatmap_familles.png" alt="Effets par famille">'])
        cols = ["Family", "MLP_Loss_Balanced_Accuracy_mean", "CNN_Loss_Balanced_Accuracy_mean",
                "MLP_Gap_Balanced_Accuracy_mean", "CNN_Gap_Balanced_Accuracy_mean"]
        for kind in ("MLP", "CNN"):
            column = kind + "_Gap_Balanced_Accuracy_mean"
            page.extend([f"<h3>Principaux effets {kind} au-delà des contrôles</h3>",
                         table(by_feature[cols].nlargest(15, column).reset_index())])
        page.extend(f'<img src="famille_{f}.png" alt="Effets pour {f}">' for f in FAMILIES)
    (args.out / "index.html").write_text("\n".join(page) + "</html>", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    paths = Paths()
    for option, key in (("mlp", "mlp_results_dir"), ("cnn", "cnn_all_results_dir"),
                        ("mlp-leace", "leace_results_dir"), ("cnn-leace", "leace_cnn_results_dir"),
                        ("out", "comparison_dir")):
        parser.add_argument("--" + option, type=Path, default=paths.location(key))
    parser.add_argument("--classification-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
