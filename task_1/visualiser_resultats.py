"""Generate the autonomous Task 1 report from existing LOSO results."""
import argparse
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

if __package__:
    from .loso_normalisation_sujet import STATS, VERSIONS, summarise
    from .paths import Paths
else:
    from loso_normalisation_sujet import STATS, VERSIONS, summarise
    from paths import Paths

FAMILIES = ["T", "F", "TF", "C", "X", "R"]
COLORS = dict(zip(FAMILIES, ["#2563eb", "#08916d", "#c58112",
                            "#8758bf", "#d55378", "#16849b"]))
COUNTS = dict(zip(FAMILIES, [10, 16, 11, 7, 5, 14]))
IDS = {f"{f}{i:03d}": f for f, n in COUNTS.items() for i in range(1, n + 1)}
VERSION_COLORS = ["#db6b1b", "#8758bf", "#16849b"]
STAT_LABELS = dict(zip(STATS, ["Mean", "Standard deviation", "Median", "Q25", "Q75", "Max"]))
METRICS = {"r2": "R²", "r": "Pearson"}
NOTICE = (
    "Nested LOSO : un sujet tenu à l’écart par pli externe, et LOSO interne pour "
    "sélectionner l’alpha Ridge de chaque cible. A : données brutes. B : cibles "
    "normalisées par sujet. C : cibles et embeddings normalisés par sujet. "
    "Analyse principale C : variations intra-sujet avec normalisation transductive "
    "utilisant l’ensemble des essais du sujet testé, sans démonstration de "
    "généralisation inductive sans calibration."
)


def load_results(source):
    config_file = source / "config.json"
    if not config_file.is_file():
        config_file = source / "config.portable.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    scores = pd.read_csv(source / "scores_par_sujet.csv")
    cache = source / "pooled_embeddings.npy"
    if cache.is_file():
        pooled = np.load(cache, mmap_mode="r", allow_pickle=False)
        if (pooled.shape != (config["n_trials"] + config["dropped_nonfinite_rows"],
                             config.get("n_embedding_dimensions", 512))
                or int((~np.isfinite(pooled).all(axis=1)).sum()) > config["dropped_nonfinite_rows"]):
            raise ValueError("Dimensions ou valeurs non finies du cache incoh?rentes.")
    ids = scores.feature.str.extract(r"^([A-Z]+\d{3})(?:_|$)")[0]
    if set(ids) != set(IDS) or not ids.map(IDS).equals(scores.family):
        raise ValueError("Identifiants ou familles incohérents.")
    versions = config["versions"]
    controls = range(config["permutations"] + 1)
    keys = ["version", "control", "subject", "feature", "dimension"]
    targets = scores[["feature", "dimension"]].drop_duplicates()
    mode = config.get("target_mode", "statistiques")
    dimensions = (sorted(d for d in scores.dimension.unique() if d.startswith("canal_"))
                  if mode == "canal" else list(STATS))
    expected_targets = {
        (feature, dim)
        for feature in scores.feature.unique()
        for dim in (["global"] if feature.startswith("R") else dimensions)
    }
    if (len(targets) != config["n_targets"] or scores.feature.nunique() != 63
            or set(targets.itertuples(index=False, name=None)) != expected_targets
            or scores.subject.nunique() != config["n_subjects"]
            or set(scores.version) != set(versions) or set(scores.control) != set(controls)
            or len(scores) != len(versions) * len(controls) * config["n_subjects"] * len(targets)
            or scores[keys].isna().any().any() or scores.duplicated(keys).any()
            or not scores.alpha.isin(config["alphas"]).all()
            or np.isinf(scores[["r2", "r"]]).any().any()
            or scores.r.abs().gt(1 + 1e-12).any() or scores.r2.gt(1 + 1e-12).any()):
        raise ValueError("Grille LOSO, alphas ou scores incohérents.")
    features, families = summarise(scores)
    for name, computed, index in [
        ("resume_caracteristiques", features, ["version", "family", "feature"]),
        ("resume_familles", families, ["version", "family"]),
    ]:
        stored = pd.read_csv(source / f"{name}.csv")
        computed["family"] = computed.family.astype(str)
        pd.testing.assert_frame_equal(stored.set_index(index).sort_index(),
                                      computed.set_index(index).sort_index(),
                                      rtol=1e-10, atol=1e-12)
    identity = pd.read_csv(source / "decodage_identite_sujet.csv").set_index("embeddings")
    identity = identity.loc[["bruts", "normalises_par_sujet"]]
    if (not identity.balanced_accuracy.between(0, 1).all()
            or not np.allclose(identity.chance, 1 / config["n_subjects"])):
        raise ValueError("Décodage ou niveau du hasard incohérent.")
    validation = None
    path = source / "validation_interne.csv"
    if path.is_file():
        data = pd.read_csv(path)
        match = ["version", "subject", "family", "feature", "dimension", "alpha"]
        validation = scores.loc[scores.control.eq(0), match + ["r2"]].merge(
            data, on=match, validate="one_to_one", how="outer", indicator=True)
        if not validation["_merge"].eq("both").all() or np.isinf(validation.validation_r2).any():
            raise ValueError("Scores de validation interne incohérents.")
    return config, scores, features, families, identity, validation


def metric_axis(ax, metric, axis="y", logarithmic=False):
    getattr(ax, f"set_{axis}label")(METRICS[metric] + (" · échelle symlog" if logarithmic else ""))
    if logarithmic:
        getattr(ax, f"set_{axis}scale")("symlog", linthresh=1)
    getattr(ax, f"axhline" if axis == "y" else "axvline")(0, color="#334155", lw=0.8)
    ax.grid(axis=axis, alpha=0.18)


def feature_plot(features, versions, metric):
    fig, axes = plt.subplots(3, 2, figsize=(17, 20), layout="constrained", sharex=True)
    column = f"{metric}_mean"
    for ax, family in zip(axes.flat, FAMILIES):
        matrix = features.loc[features.family.eq(family)].pivot(
            index="feature", columns="version", values=column)
        y = np.arange(len(matrix))
        primary = versions[-1]
        ax.barh(y, matrix[primary], color=COLORS[family], label=primary)
        for version in versions[:-1]:
            color = VERSION_COLORS[list(VERSIONS).index(version)]
            ax.scatter(matrix[version], y, color=color, s=27,
                       marker="o" if version == "A" else "D", zorder=3, label=version)
        if len(versions) == 1:
            for i, value in enumerate(matrix[primary]):
                if np.isfinite(value):
                    ax.annotate(f"{value:.3f}", (value, i), xytext=(3, 0),
                                textcoords="offset points", va="center", fontsize=8)
        ax.set_yticks(y, matrix.index, fontsize=9)
        ax.invert_yaxis()
        metric_axis(ax, metric, "x", metric == "r2" and len(versions) > 1)
        ax.set_title(f"{family} — {len(matrix)} caractéristiques")
        ax.legend(fontsize=8)
    fig.suptitle(f"REVE — 63 caractéristiques · {' / '.join(versions)}", fontsize=17)
    return fig


def boxes(features, primary, metric):
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    rng = np.random.default_rng(42)
    for i, family in enumerate(FAMILIES):
        vals = features.loc[features.family.eq(family), f"{metric}_mean"].dropna()
        ax.boxplot([vals], positions=[i], widths=0.45, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": COLORS[family], "alpha": 0.25},
                   medianprops={"color": "black"})
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, color=COLORS[family],
                   s=35, zorder=3)
        ax.scatter(i, vals.mean(), marker="D", color="black", s=35, zorder=4)
    ax.set_xticks(range(len(FAMILIES)), FAMILIES)
    metric_axis(ax, metric, logarithmic=metric == "r2" and primary == "A")
    ax.set_title(f"Version {primary} · distribution entre caractéristiques · losange noir = moyenne")
    return fig


def grouped(ax, table, categories, versions, column, family_palette=False):
    x = np.arange(len(categories))
    width = 0.78 / len(versions)
    for i, version in enumerate(versions):
        block = table.loc[table.version.eq(version)].reindex(categories)
        style = list(VERSIONS).index(version)
        colors = [COLORS[f] for f in categories] if family_palette else VERSION_COLORS[style]
        ax.bar(x + (i - (len(versions) - 1) / 2) * width, block[column], width,
               color=colors, alpha=[0.42, 0.68, 1][style] if family_palette else 1,
               hatch=["//", "..", ""][style] if family_palette else None, label=version)
    ax.set_xticks(x, categories, rotation=45 if len(categories) > 6 else 0, ha="center")
    ax.legend(fontsize=8)


def heatmap(matrix, title, xlabel, metric):
    vals = matrix.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    low = min(-0.01, finite.min()) if finite.size else -0.01
    high = max(0.01, finite.max()) if finite.size else 0.01
    fig, ax = plt.subplots(figsize=(14, max(3, len(matrix) * 0.38 + 1.5)), layout="constrained")
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#cbd5e1")
    image = ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=cmap,
                      norm=TwoSlopeNorm(vmin=low, vcenter=0, vmax=high))
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=8)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns,
                  rotation=90 if len(matrix.columns) > 12 else 0, fontsize=8)
    ax.set(xlabel=xlabel, title=title)
    fig.colorbar(image, ax=ax, label=METRICS[metric])
    return fig


def validation_plot(primary, features, validation):
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    if validation is None:
        table = features.query("version == @primary")
        xcol, ycol, xlabel = "r_mean", "r2_mean", "Pearson moyen"
        title = f"Version {primary} · Pearson et R² du test externe"
        caption = "Les scores internes sont absents de cette exécution. Un point par caractéristique."
        name = "03_pearson_r2"
    else:
        frame = validation.query("version == @primary")
        table = (frame.groupby(["subject", "family", "feature"])[["validation_r2", "r2"]].mean()
                 .groupby(["family", "feature"]).mean().reset_index())
        xcol, ycol, xlabel = "validation_r2", "r2", "R² moyen de validation interne"
        title = f"Version {primary} · validation interne et test externe"
        caption = "Validation au meilleur alpha, sélectionné par cible dans chaque pli externe."
        name = "03_validation_test"
        lo = min(0, table[[xcol, ycol]].min().min())
        ax.plot([lo, 1], [lo, 1], "--", color="#64748b", linewidth=1)
        if primary == "A":
            ax.set_xscale("symlog", linthresh=1)
            xlabel += " · échelle symlog"
    for family in FAMILIES:
        group = table.loc[table.family.eq(family)]
        ax.scatter(group[xcol], group[ycol], color=COLORS[family], label=family, s=45, alpha=0.85)
        for row in group.loc[group[ycol].lt(0)].itertuples(index=False):
            ax.annotate(row.feature, (getattr(row, xcol), getattr(row, ycol)),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    metric_axis(ax, "r2", logarithmic=primary == "A")
    ax.axvline(0, color="#334155", lw=0.7)
    ax.set(xlabel=xlabel, title=title)
    ax.legend(title="Famille")
    return fig, name, title, caption


def permutation_plot(table, version, item, categories):
    data = table.loc[table.version.eq(version)]
    matrix = data.pivot(index=item, columns="control", values="r").reindex(categories)
    null = matrix.drop(columns=0)
    wide = item == "feature"
    fig, ax = plt.subplots(figsize=(16, 19) if wide else (11, 6), layout="constrained")
    x = np.arange(len(categories))
    colors = [COLORS[IDS[c.split("_")[0]] if wide else c] for c in categories]

    def points(position, score, **options):
        ax.scatter(score, position, **options) if wide else ax.scatter(position, score, **options)

    points(x, matrix[0], c=colors, marker="o", label="Réel", zorder=3)
    points(x, null.mean(axis=1), color="#db6b1b", marker="D", s=22, label="Moyenne permutée")
    points(x, null.max(axis=1), color="black", marker="|" if wide else "_",
           s=90, label="Maximum permuté")
    for j, control in enumerate(null):
        offset = 0.15 * (j - (len(null.columns) - 1) / 2) / max(1, len(null.columns) - 1)
        points(x + offset, null[control], color="#64748b", alpha=0.55, s=15,
               label="Permutations individuelles" if j == 0 else None)
    if wide:
        ax.set_yticks(x, categories, fontsize=8)
        ax.invert_yaxis()
    else:
        ax.set_xticks(x, categories)
    metric_axis(ax, "r", "x" if wide else "y")
    ax.set_title(f"Version {version} · contrôle par {len(null.columns)} permutations")
    ax.legend(fontsize=8)
    return fig


def dashboard(out, charts, tables, protocol):
    parts = [
        '<!doctype html><html lang="fr"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>REVE — résultats Nested LOSO</title>",
        """<style>
        body { font: 16px system-ui; background: #f1f5f9; color: #142033; margin: 0; }
        main { max-width: 1400px; margin: auto; padding: 32px; }
        section { background: white; border-radius: 14px; padding: 24px; margin: 24px 0; }
        img { width: 100%; height: auto; }
        table { border-collapse: collapse; font-size: 14px; }
        td, th { padding: 8px 12px; border-bottom: 1px solid #ddd; text-align: right; }
        p { line-height: 1.6; }
        .scroll { overflow: auto; }
        nav { margin: 20px 0; }
        summary { cursor: pointer; }
        </style></head><body>""",
        "<main><h1>REVE — résultats des six familles</h1>",
        f"<p>{html.escape(NOTICE)}</p><p>{html.escape(protocol)}</p>",
        '<nav aria-label="Sommaire"><details><summary>Sommaire des figures</summary><ul>',
    ]
    parts.extend(f'<li><a href="#{name}">{html.escape(title)}</a></li>'
                 for name, title, _ in charts)
    parts.append("</ul></details></nav>")

    def section_table(title, table):
        return (f'<section><h2>{html.escape(title)}</h2><div class="scroll">'
                f'{table.to_html(index=False, escape=True, float_format=lambda x: f"{x:.5g}")}'
                '</div></section>')

    parts.append(section_table(*tables[0]))
    for name, title, caption in charts:
        encoded = base64.b64encode((out / f"{name}.png").read_bytes()).decode("ascii")
        parts.append(f'<section id="{name}"><h2>{html.escape(title)}</h2>'
                     f'<p>{html.escape(caption)}</p><img alt="{html.escape(title)}" '
                     f'src="data:image/png;base64,{encoded}"></section>')
    parts.extend(section_table(title, data) for title, data in tables[1:])
    parts.append("</main></body></html>")
    (out / "dashboard.html").write_text("\n".join(parts), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true", help="Vérifier les résultats seuls")
    args = parser.parse_args(argv)
    paths = Paths(args.config)
    source = (args.results or paths.location("results_dir")).resolve()
    out = (args.out or paths.location("visualisations_dir")).resolve()
    config, scores, features, families, identity, validation = load_results(source)
    print(f"[OK] {len(scores):,} scores · {features.feature.nunique()} caractéristiques "
          f"· {config['n_subjects']} sujets · versions {', '.join(config['versions'])}", flush=True)
    if args.check:
        return 0
    if source.is_relative_to(out) or out.is_relative_to(source):
        parser.error("--out doit être distinct du dossier des scores.")
    if config.get("target_mode") == "canal" and out == paths.location("visualisations_dir"):
        parser.error("Les figures par canal exigent un --out distinct du rapport principal.")
    out.mkdir(parents=True, exist_ok=True)
    versions = [v for v in VERSIONS if v in config["versions"]]
    primary = versions[-1]
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    charts = []

    def save(fig, name, title, caption):
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        charts.append((name, title, caption))
        print(f"  Figure créée : {name}", flush=True)

    for metric, label in METRICS.items():
        save(feature_plot(features, [primary], metric),
             "01_features" if metric == "r2" else "01_features_pearson",
             f"Les 63 caractéristiques · {label} · {primary}",
             "Score moyen entre sujets après agrégation des dimensions par caractéristique.")
    selected = features.query("version == @primary")
    for metric, label in METRICS.items():
        save(boxes(selected, primary, metric),
             "02_familles" if metric == "r2" else "02_familles_pearson",
             f"Comparaison des familles · {label} · {primary}",
             "Point : caractéristique. Trait noir : médiane. Losange noir : moyenne des caractéristiques.")
    save(*validation_plot(primary, features, validation))

    connectivity = features.query("family == 'R'").set_index("feature")
    connectivity.to_csv(out / "comparaison_R.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout="constrained")
    for (metric, label), ax in zip(METRICS.items(), axes):
        grouped(ax, connectivity, sorted(set(connectivity.index)), versions, f"{metric}_mean")
        metric_axis(ax, metric, logarithmic=metric == "r2" and "A" in versions)
    fig.suptitle("R — 14 caractéristiques de connectivité")
    save(fig, "04_R_comparaison", "Comparaison de la connectivité R",
         "Moyennes des scores externes pour les cibles globales.")

    feature_subject = (scores.groupby(["version", "control", "subject", "family", "feature"])[["r2", "r"]]
                       .mean().reset_index())
    family_subject = (feature_subject.groupby(["version", "control", "subject", "family"])[["r2", "r"]]
                      .mean().reset_index())
    real = scores.query("control == 0")
    feature_real = feature_subject.query("control == 0")
    family_real = family_subject.query("control == 0").drop(columns="control")
    primary_dims = real.query("version == @primary")
    channel_mode = config.get("target_mode") == "canal"
    dimensions = (sorted(d for d in primary_dims.dimension.unique() if d.startswith("canal_"))
                  if channel_mode else list(STATS))
    dimension_name = "canaux" if channel_mode else "statistiques"
    dimension_axis = "Indice du canal EEG (base 0)" if channel_mode else "Statistique sur les canaux"
    heat_caption = "Bleu : positif. Rouge : négatif. Gris : indéfini. Échelle propre à chaque figure."
    for family in FAMILIES:
        for metric, label in METRICS.items():
            matrix = (primary_dims.query("family == @family")
                      .groupby(["feature", "dimension"])[metric].mean().unstack())
            matrix = matrix.reindex(columns=["global"] if family == "R" else dimensions)
            matrix = matrix.rename(columns=STAT_LABELS)
            axis = "Cible globale" if family == "R" else dimension_axis
            title = f"{family} · {label} · {primary} · {axis}"
            save(heatmap(matrix, title, axis, metric),
                 f"05_{dimension_name}_{family}_{metric}", title, heat_caption)
            by_subject = feature_real.query("version == @primary and family == @family")
            matrix = by_subject.pivot(index="feature", columns="subject", values=metric)
            title = f"{family} · {label} · {primary} · caractéristiques × sujets"
            save(heatmap(matrix, title, "Sujet du pli externe", metric),
                 f"05_sujets_{family}_{metric}", title, heat_caption)

    for metric, label in METRICS.items():
        save(feature_plot(features, versions, metric), f"06_features_abc_{metric}",
             f"Comparaison des versions par caractéristique · {label}",
             "L’échelle symlog de la comparaison R² est linéaire autour de zéro et conserve les valeurs extrêmes.")
    family_index = families.set_index("family")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    dispersion = family_real.groupby(["version", "family"]).r2.std().rename("r2_std")
    comparison = families.join(dispersion, on=["version", "family"]).set_index("family")
    for ax, column, label, metric in zip(
        axes.flat,
        ["r2_mean", "r2_median", "r_mean", "r_std_between_subjects", "r2_std"],
        ["R² moyen", "R² médian", "Pearson moyen", "Écart-type de Pearson entre sujets",
         "Écart-type de R² entre sujets"],
        ["r2", "r2", "r", "r", "r2"],
    ):
        grouped(ax, comparison, FAMILIES, versions, column, family_palette=True)
        metric_axis(ax, metric, logarithmic=metric == "r2" and "A" in versions)
        ax.set_title(label)
    axes.flat[-1].set_visible(False)
    save(fig, "06_familles_abc", "Comparaison des versions par famille",
         "Les résumés officiels agrègent d’abord les cibles et les caractéristiques dans chaque sujet.")

    perm_features = feature_subject.groupby(["version", "family", "feature", "control"]).r.mean().reset_index()
    perm_families = family_subject.groupby(["version", "family", "control"]).r.mean().reset_index()
    if config["permutations"]:
        feature_order = sorted(features.feature.unique(), key=lambda f: (FAMILIES.index(IDS[f.split("_")[0]]), f))
        for version in versions:
            for table, item, categories in [(perm_families, "family", FAMILIES),
                                             (perm_features, "feature", feature_order)]:
                save(permutation_plot(table, version, item, categories), f"07_{item}_{version}",
                     f"Permutations · {version} · {'familles' if item == 'family' else 'caractéristiques'}",
                     "Contrôle descriptif. Les permutations intra-sujet ne permettent ici qu’une résolution limitée.")

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    values = identity.balanced_accuracy * 100
    bars = ax.bar(["Embeddings bruts", "Normalisés par sujet"], values,
                  color=[VERSION_COLORS[0], VERSION_COLORS[2]], width=0.5)
    ax.bar_label(bars, labels=[f"{v:.1f} %" for v in values], padding=4)
    chance = 100 * identity.chance.iloc[0]
    ax.axhline(chance, color="#334155", linestyle="--", label=f"Hasard · {chance:g} %")
    ax.set(ylim=(0, 110), ylabel="Balanced accuracy (%)", title="Décodage de l’identité du sujet")
    ax.legend()
    save(fig, "08_identite", "Décodage de l’identité du sujet",
         "Régression logistique, validation croisée stratifiée à cinq plis. Le score après normalisation ne démontre pas une suppression complète de l’identité.")

    for metric, label in METRICS.items():
        fig, axes = plt.subplots(3, 2, figsize=(15, 11), layout="constrained")
        subjects = sorted(family_real.subject.unique())
        for family, ax in zip(FAMILIES, axes.flat):
            table = family_real.query("family == @family").set_index("subject")
            grouped(ax, table, subjects, versions, metric)
            metric_axis(ax, metric, logarithmic=metric == "r2" and "A" in versions)
            ax.set_title(family, color=COLORS[family])
        fig.suptitle(f"{label} par sujet et famille · comparaison des versions")
        save(fig, f"09_sujets_{metric}", f"{label} par sujet et famille",
             "Un groupe de barres par sujet tenu à l’écart. Les différences de hauteur décrivent la variabilité entre sujets.")

    null_summary = (perm_features.query("control > 0").groupby(["version", "family", "feature"]).r
                    .max().rename("r_shuffle_max"))
    feature_details = features.join(null_summary, on=["version", "family", "feature"])
    tables = [
        ("Résumé par famille", family_index.reset_index()),
        ("Tableau des 63 caractéristiques", feature_details),
        ("Scores par sujet et famille", family_real),
        ("Scores par sujet et caractéristique", feature_real.drop(columns="control")),
        ("Pearson réel et permutations par famille", perm_families),
        ("Pearson réel et permutations par caractéristique", perm_features),
        ("Décodage de l’identité du sujet", identity.reset_index()),
    ]
    if validation is not None:
        tables.append(("Validation interne par cible au meilleur alpha", validation.drop(columns="_merge")))
    for name, table in [("resultats_63_features", feature_details), ("resume_familles", families),
                        ("familles_par_sujet", family_real), ("caracteristiques_par_sujet", feature_real),
                        ("permutations_familles", perm_families), ("permutations_features", perm_features)]:
        table.to_csv(out / f"{name}.csv", index=False)
    protocol = (f"{config['n_trials']} essais · {config['n_subjects']} sujets · "
                f"63 caractéristiques · {config['n_targets']} cibles · "
                f"{config.get('n_embedding_dimensions', 512)} dimensions d’embedding après mean pooling · "
                f"{config['dropped_nonfinite_rows']} lignes non finies retirées · "
                f"{config['permutations']} permutations par version. Cibles : {config['targets']}.")
    dashboard(out, charts, tables, protocol)
    (out / "sources.json").write_text(json.dumps({
        "results": str(args.results or paths.settings["results_dir"]),
        "validation_interne": validation is not None,
        "figures": [name for name, _, _ in charts],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Tableau de bord : {out / 'dashboard.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
