"""Figures et tableau de bord hors ligne des effacements LEACE du MLP."""
import argparse
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

METRICS = {
    "Balanced_Accuracy": ("Balanced accuracy", -100),
    "Accuracy": ("Accuracy", -100),
    "Macro_F1": ("Macro-F1", -100),
    "Log_Loss": ("Log-loss", 1),
}
FAMILIES = ["T", "F", "TF", "C", "R", "X"]


def load_results(folder):
    raw = pd.read_csv(folder / "by_subject.csv")
    keys = ["Feature", "Subject", "Control"]
    if raw[keys].isna().any().any() or raw.duplicated(keys).any():
        raise ValueError("Identifiants manquants ou dupliqués dans by_subject.csv")
    expected = pd.MultiIndex.from_product([raw[k].unique() for k in keys], names=keys)
    if (len(expected) != len(raw) or raw.Feature.nunique() != 63
            or raw.Subject.nunique() != 10 or raw.Control.min() != 0 or raw.Control.max() < 1):
        raise ValueError("Résultats incomplets : vérifier les 63 features, 10 sujets et contrôles.")
    paired = raw.set_index(keys).xs(0, level="Control").sort_index().copy()
    random = raw[raw.Control > 0].groupby(keys[:2]).mean(numeric_only=True)
    for metric, (_, factor) in METRICS.items():
        delta = "Delta_" + metric
        if not np.isfinite(raw[[metric, delta, "Original_" + metric]]).all().all():
            raise ValueError(f"Score non fini : {metric}")
        paired["Random_" + metric] = random[metric]
        paired["Loss_" + metric] = paired[delta] * factor
        paired["Random_loss_" + metric] = random[delta] * factor
        paired["Gap_" + metric] = paired["Loss_" + metric] - paired["Random_loss_" + metric]
    paired["Random_RMS"] = random.Test_RMS_change
    paired["Ratio_RMS"] = paired.Test_RMS_change / paired.Random_RMS.replace(0, np.nan)
    return paired, raw.Control.nunique() - 1


def plot_family(pdf, folder, family, data):
    metric = "Balanced_Accuracy"
    summary = data.groupby(level="Feature").mean().sort_values("Gap_" + metric, ascending=False)
    names = summary.index
    matrix = data["Gap_" + metric].unstack().loc[names]
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(15, max(4.8, 2.3 + .43 * len(names))),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=.21, right=.96, top=.82, bottom=.16, wspace=.19)
    y = np.arange(len(names))
    for prefix, offset, color, label in [
            ("Loss_", -.13, "#146653", "LEACE"), ("Random_loss_", .13, "#bb651f", "Aléatoire")]:
        ax.scatter(summary[prefix + metric], y + offset, color=color, label=label)
    ax.axvline(0, color="gray", lw=1)
    ax.set(yticks=y, yticklabels=names, xlabel="Perte moyenne (points)", ylim=(len(y)-.5, -.5))
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False)
    limit = max(1., np.abs(matrix.values).max())
    heat.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    heat.set(yticks=[], xticks=np.arange(len(matrix.columns)),
             xticklabels=[s.removeprefix("sub-") for s in matrix.columns],
             xlabel="Sujet tenu à l’écart", title="Perte supplémentaire de LEACE (points)")
    for (i, j), value in np.ndenumerate(matrix.values):
        heat.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8,
                  color="white" if abs(value) > .58 * limit else "black")
    fig.suptitle(f"Famille {family}   |   Balanced accuracy", fontsize=16)
    fig.savefig(folder / f"{family}.png", dpi=145)
    pdf.savefig(fig)
    plt.close(fig)


def table(frame):
    return frame.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--results", type=Path, default=root / "results_leace_mlp")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    data, n = load_results(args.results)
    out = args.out or args.results / "visualisations"
    out.mkdir(parents=True, exist_ok=True)
    family_ids = data.index.get_level_values("Feature").str.extract(r"^([A-Z]+)\d", expand=False)
    if set(family_ids) != set(FAMILIES):
        raise ValueError("Familles EEG inattendues dans by_subject.csv")
    page = ["""<!doctype html><html lang="fr"><meta charset="utf-8"><title>Effacement LEACE</title>
<style>body{font:16px system-ui;color:#17313a;max-width:1350px;margin:auto;padding:24px}
img{width:100%}details{margin:10px 0}summary{cursor:pointer}section{overflow-x:auto}
td,th{padding:5px 9px;text-align:right;border-bottom:1px solid #ddd}table{border-collapse:collapse}
nav a{margin-right:18px}</style><h1>Effacement LEACE sur le MLP</h1>""",
            f"<p>63 caractéristiques · 10 sujets · {n} contrôles aléatoires par effacement. "
            "Chaque sujet a le même poids. Une perte positive indique une dégradation. "
            "La carte montre la perte ciblée moins la perte aléatoire. Son échelle de couleur "
            "est propre à chaque famille.</p><nav>" +
            "".join(f'<a href="#{f}">{f}</a>' for f in FAMILIES) + "</nav>"]
    with PdfPages(out / "figures_leace.pdf") as pdf:
        for family in FAMILIES:
            subset = data[family_ids == family]
            plot_family(pdf, out, family, subset)
            page.append(f'<section id="{family}"><h2>Famille {family}</h2><img src="{family}.png" alt="Pertes par feature et par sujet">')
            for metric, (label, factor) in METRICS.items():
                cols = ["Original_"+metric, metric, "Random_"+metric, "Gap_"+metric]
                means = subset[cols].groupby(level="Feature").mean().sort_values(cols[-1], ascending=False)
                means.iloc[:, :3] *= abs(factor)
                means.columns = ["Original", "LEACE", "Aléatoire", "Perte supplémentaire"]
                unit = "échelle 0–100, écarts en points" if factor < 0 else "unités de log-loss"
                page.append(f"<details><summary>{label} ({unit})</summary>{table(means.reset_index())}</details>")
            page.append("<h3>Détail par caractéristique</h3><p>Le ratio RMS compare les amplitudes ciblée et aléatoire. "
                        "Les scores de sonde évaluent la prédiction de la feature chez le sujet tenu à l’écart "
                        "par rapport à la moyenne d’entraînement. Un tiret indique une valeur indéfinie.</p>")
            for feature, group in subset.groupby(level="Feature"):
                detail = group.droplevel("Feature")[["Loss_Balanced_Accuracy", "Random_loss_Balanced_Accuracy",
                        "Gap_Balanced_Accuracy", "Ratio_RMS", "Rank", "Probe_skill_original", "Probe_skill_erased"]].copy()
                detail.columns = ["Perte LEACE (pts)", "Perte aléatoire (pts)", "Écart (pts)", "Ratio RMS",
                                  "Rang", "Sonde avant", "Sonde après"]
                page.append(f"<details><summary>{escape(feature)}</summary>{table(detail.reset_index())}</details>")
            page.append("</section>")
    data.to_csv(out / "comparaisons_par_sujet.csv")
    (out / "index.html").write_text("\n".join(page) + "</html>", encoding="utf-8")
    print((out / "index.html").resolve())


if __name__ == "__main__":
    main()
