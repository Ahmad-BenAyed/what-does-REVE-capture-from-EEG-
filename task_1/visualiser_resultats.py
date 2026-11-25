"""Visualiser les résultats des six familles de features REVE."""
import argparse
import base64
import html
import json
from pathlib import Path

if __package__:
    from .paths import Paths, prepare_output, ridge_csv
else:
    from paths import Paths, prepare_output, ridge_csv

import matplotlib
matplotlib.use("Agg")  # Compatible avec Slurm, sans écran graphique.
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


FAMILIES = ["T", "F", "TF", "C", "X", "R"]

COLORS = dict(zip(
    FAMILIES,
    ["#2563eb", "#08916d", "#c58112",
     "#8758bf", "#d55378", "#16849b"],
))

SPECS = {
    "T": (
        "reve_temporal_channel_results.csv",
        10,
    ),
    "F": (
        "reve_spectral_channel_results.csv",
        16,
    ),
    "TF": (
        "reve_tf_channel_results.csv",
        11,
    ),
    "C": (
        "reve_c_channel_results.csv",
        7,
    ),
    "X": (
        "reve_x_channel_results.csv",
        5,
    ),
    "R": (
        "reve_r_dimensions_results.csv",
        14,
    ),
}

NOTICE = (
    "R² bruts de régression. Partition aléatoire par essai : "
    "des sujets peuvent être partagés. Ces scores ne mesurent "
    "ni une accuracy de classification ni un effet causal."
)


def read_result(path, family, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Fichier manquant : {path}")

    data = pd.read_csv(path)

    required = {
        "Feature", "Best_Alpha", "Validation_R2", "Test_R2"
    }

    if not required.issubset(data.columns):
        raise ValueError(
            f"Colonnes manquantes dans {path} : "
            f"{required - set(data.columns)}"
        )

    data["Feature_ID"] = (
        data["Feature"].astype(str)
        .str.extract(r"^([A-Z]+\d{3})")[0]
    )

    expected_ids = {
        f"{family}{j:03d}"
        for j in range(1, expected + 1)
    }

    if (
        len(data) != expected
        or data["Feature_ID"].duplicated().any()
        or set(data["Feature_ID"]) != expected_ids
    ):
        raise ValueError(
            f"Features incohérentes pour {family} : {path}"
        )

    for column in ["Validation_R2", "Test_R2"]:
        data[column] = pd.to_numeric(
            data[column], errors="raise"
        )

        if np.isinf(data[column]).any():
            raise ValueError(f"Score infini dans {path}")

    data["Family"] = family

    return data.sort_values("Feature_ID").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--r-mode",
        choices=["global", "summary"],
        default="global",
    )
    args = parser.parse_args()

    infrastructure = Paths(args.config)
    root = args.root or infrastructure.location("family_dir")
    out = args.out or infrastructure.location("visualisations_dir")

    tables = []
    paths = {}

    print("Chargement des résultats...", flush=True)

    for family in FAMILIES:
        _, expected = SPECS[family]
        folder = infrastructure.families[family]

        if family == "R" and args.r_mode == "summary":
            folder = "results_r_summary"

        path = root / family / "resultats" / folder / ridge_csv(family)
        data = read_result(path, family, expected)
        data["Source"] = str(path.resolve())

        tables.append(data)
        paths[family] = path

        print(
            f"  {family} : {len(data)} features",
            flush=True,
        )

    all_results = pd.concat(tables, ignore_index=True)

    if len(all_results) != 63:
        raise ValueError("Le tableau doit contenir 63 features.")

    prepare_output(out)

    all_results.to_csv(
        out / "resultats_63_features.csv",
        index=False,
    )

    summary = all_results.groupby(
        "Family", sort=False
    ).agg(
        n_features=("Feature", "size"),
        n_scores_defined=("Test_R2", "count"),
        mean_test_r2=("Test_R2", "mean"),
        median_test_r2=("Test_R2", "median"),
        min_test_r2=("Test_R2", "min"),
        max_test_r2=("Test_R2", "max"),
        n_negative=("Test_R2", lambda v: int((v < 0).sum())),
    )

    summary.to_csv(out / "resume_familles.csv")

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    charts = []

    def save(fig, name, title, caption):
        fig.savefig(
            out / (name + ".png"),
            dpi=160,
            bbox_inches="tight",
        )
        fig.savefig(
            out / (name + ".pdf"),
            bbox_inches="tight",
        )
        plt.close(fig)
        charts.append((name, title, caption))
        print(f"  Figure créée : {name}", flush=True)

    # --------------------------------------------------------
    # 1. Les 63 features : validation et test.
    # --------------------------------------------------------
    values = all_results[
        ["Validation_R2", "Test_R2"]
    ].to_numpy().ravel()
    values = values[np.isfinite(values)]

    low = min(0.0, values.min()) if len(values) else 0.0
    high = max(1.0, values.max()) if len(values) else 1.0
    pad = 0.05 * max(1.0, high - low)

    fig, axes = plt.subplots(
        3, 2,
        figsize=(17, 20),
        layout="constrained",
    )

    for ax, family in zip(axes.ravel(), FAMILIES):
        data = all_results[
            all_results["Family"] == family
        ]
        y = np.arange(len(data))

        ax.barh(
            y, data["Test_R2"],
            color=COLORS[family],
            label="Test",
        )

        ax.scatter(
            data["Validation_R2"], y,
            color="#db6b1b",
            s=22,
            zorder=3,
            label="Validation",
        )

        labels = data["Feature"].str.replace(
            "_", " ", regex=False
        )

        ax.set_yticks(y, labels=labels, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#334155", linewidth=0.8)
        ax.set_xlim(low - pad, high + pad)
        ax.set_title(f"{family} — {len(data)} features")
        ax.set_xlabel("R² brut")
        ax.grid(axis="x", alpha=0.18)
        ax.legend(fontsize=8, loc="best")

        for i, value in enumerate(data["Test_R2"]):
            if np.isfinite(value):
                ax.text(
                    value, i, f" {value:.3f} ",
                    va="center", ha="left", fontsize=8,
                )
            else:
                ax.text(
                    0, i, " indéfini",
                    va="center", fontsize=8,
                )

    fig.suptitle(
        f"REVE — 63 features · R : {args.r_mode}\n"
        "Partition aléatoire par essai, sans séparation des sujets",
        fontsize=17,
    )

    save(
        fig,
        "01_features",
        "Les 63 features",
        "Barres : test ; points : validation. "
        "Les R² négatifs sont conservés.",
    )

    # --------------------------------------------------------
    # 2. Distribution des scores par famille.
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(11, 6),
        layout="constrained",
    )
    rng = np.random.default_rng(42)

    for i, family in enumerate(FAMILIES):
        scores = all_results.loc[
            all_results["Family"] == family, "Test_R2"
        ].dropna().to_numpy()

        if not len(scores):
            continue

        ax.boxplot(
            [scores],
            positions=[i],
            widths=0.45,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": COLORS[family],
                "alpha": 0.25,
            },
            medianprops={"color": "black"},
        )

        ax.scatter(
            i + rng.uniform(-0.12, 0.12, len(scores)),
            scores,
            color=COLORS[family],
            s=35,
            zorder=3,
        )

        ax.scatter(
            [i], [scores.mean()],
            marker="D",
            color="black",
            s=35,
            zorder=4,
        )

    ax.set_xticks(range(6), labels=FAMILIES)
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_ylabel("R² test brut")
    ax.set_title(
        "Distribution entre features · losange noir = moyenne"
    )
    ax.grid(axis="y", alpha=0.18)

    save(
        fig,
        "02_familles",
        "Comparaison des familles",
        "Chaque point représente une feature. "
        "Les boîtes décrivent une dispersion entre features, "
        "pas un intervalle de confiance entre sujets.",
    )

    # --------------------------------------------------------
    # 3. Validation contre test.
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(8, 7),
        layout="constrained",
    )

    for family in FAMILIES:
        data = all_results[
            all_results["Family"] == family
        ]

        ax.scatter(
            data["Validation_R2"],
            data["Test_R2"],
            color=COLORS[family],
            label=family,
            s=45,
            alpha=0.85,
        )

        for _, row in data.iterrows():
            if (
                np.isfinite(row["Test_R2"])
                and row["Test_R2"] < 0
                and np.isfinite(row["Validation_R2"])
            ):
                ax.annotate(
                    row["Feature_ID"],
                    (row["Validation_R2"], row["Test_R2"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.plot(
        [low - pad, high + pad],
        [low - pad, high + pad],
        "--",
        color="#64748b",
        linewidth=1,
    )

    ax.axhline(0, color="#334155", linewidth=0.7)
    ax.axvline(0, color="#334155", linewidth=0.7)
    ax.set(
        xlabel="R² validation brut",
        ylabel="R² test brut",
        title="Validation et test",
        xlim=(low - pad, high + pad),
        ylim=(low - pad, high + pad),
    )
    ax.legend(title="Famille")
    ax.grid(alpha=0.15)

    save(
        fig,
        "03_validation_test",
        "Validation et test",
        "La diagonale indique des scores égaux. "
        "Le test ne sert pas à sélectionner l'alpha.",
    )

    # --------------------------------------------------------
    # 4. Comparaison R global / R summary.
    # --------------------------------------------------------
    base = root / "R" / "resultats"

    global_path = (
        base / infrastructure.families["R"] / "reve_r_ridge_results.csv"
    )
    summary_path = (
        base / "results_r_summary" / "reve_r_ridge_results.csv"
    )

    if global_path.is_file() and summary_path.is_file():
        global_r = read_result(global_path, "R", 14)
        summary_r = read_result(summary_path, "R", 14)

        comparison = global_r[
            ["Feature_ID", "Feature", "Test_R2"]
        ].merge(
            summary_r[["Feature_ID", "Test_R2"]],
            on="Feature_ID",
            suffixes=("_global", "_summary"),
            validate="one_to_one",
        )

        comparison.to_csv(
            out / "comparaison_R.csv",
            index=False,
        )

        fig, ax = plt.subplots(
            figsize=(12, 6),
            layout="constrained",
        )
        x = np.arange(14)

        ax.bar(
            x - 0.19,
            comparison["Test_R2_global"],
            width=0.38,
            color="#16849b",
            label="Global : 14 dimensions",
        )
        ax.bar(
            x + 0.19,
            comparison["Test_R2_summary"],
            width=0.38,
            color="#8b5fbf",
            label="Summary : 54 dimensions",
        )

        ax.set_xticks(
            x,
            labels=comparison["Feature_ID"],
            rotation=45,
            ha="right",
        )
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.set(
            ylabel="R² test brut",
            title="R — deux représentations des cibles",
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.18)

        save(
            fig,
            "04_R_comparaison",
            "R : global et summary",
            "Pour R005–R014, les cibles diffèrent : "
            "moyenne des paires contre cinq statistiques. "
            "Il ne s'agit pas de deux scores sur la même cible.",
        )

    # --------------------------------------------------------
    # 5. Cartes de scores par canal.
    # --------------------------------------------------------
    for family in FAMILIES[:-1]:
        detail_path = paths[family].with_name(
            SPECS[family][0]
        )

        if not detail_path.is_file():
            print(
                f"Détail par canal absent : {detail_path}",
                flush=True,
            )
            continue

        data = pd.read_csv(detail_path)
        required = {"Feature", "Channel", "Test_R2"}

        if not required.issubset(data.columns):
            raise ValueError(
                f"Format par canal inattendu : {detail_path}"
            )

        data["Feature_ID"] = (
            data["Feature"]
            .str.extract(r"^([A-Z]+\d{3})")[0]
        )

        if data.duplicated(["Feature_ID", "Channel"]).any():
            raise ValueError(
                f"Lignes par canal dupliquées : {detail_path}"
            )

        matrix = data.pivot(
            index="Feature_ID",
            columns="Channel",
            values="Test_R2",
        ).sort_index()

        values = matrix.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]

        if not len(finite_values):
            continue

        norm = TwoSlopeNorm(
            vmin=min(-0.01, float(finite_values.min())),
            vcenter=0,
            vmax=max(0.01, float(finite_values.max())),
        )

        fig, ax = plt.subplots(
            figsize=(14, max(3, len(matrix) * 0.38 + 1.5)),
            layout="constrained",
        )

        cmap = plt.get_cmap("RdBu").copy()
        cmap.set_bad("#cbd5e1")

        image = ax.imshow(
            np.ma.masked_invalid(values),
            aspect="auto",
            cmap=cmap,
            norm=norm,
        )

        ax.set_yticks(
            range(len(matrix)),
            labels=matrix.index,
        )
        ax.set_xticks(
            range(len(matrix.columns)),
            labels=matrix.columns,
            rotation=90,
            fontsize=8,
        )
        ax.set(
            xlabel="Indice du canal dans les fichiers EEG",
            title=f"{family} — R² test par feature et canal",
        )

        fig.colorbar(image, ax=ax, label="R² brut")

        save(
            fig,
            f"05_canaux_{family}",
            f"{family} : détail par canal",
            "Bleu : positif ; rouge : négatif ; gris : indéfini. "
            "Échelle propre à cette figure, sans tronquer "
            "les scores extrêmes.",
        )

    # --------------------------------------------------------
    # 6. Tableau de bord HTML autonome.
    # --------------------------------------------------------
    parts = [
        '<!doctype html><html lang="fr">',
        '<meta charset="utf-8">',
        "<title>REVE — résultats des six familles</title>",
        """<style>
        body {
            font: 16px system-ui;
            background: #f1f5f9;
            color: #142033;
            margin: 0;
        }
        main {
            max-width: 1400px;
            margin: auto;
            padding: 32px;
        }
        section {
            background: white;
            border-radius: 14px;
            padding: 24px;
            margin: 24px 0;
        }
        img { width: 100%; height: auto; }
        table { border-collapse: collapse; font-size: 14px; }
        td, th {
            padding: 8px 12px;
            border-bottom: 1px solid #ddd;
            text-align: right;
        }
        p { line-height: 1.6; }
        .scroll { overflow: auto; }
        </style>""",
        "<main><h1>REVE — résultats des six familles</h1>",
        f"<p>{html.escape(NOTICE)}</p>",
        f"<p>63 features ; mode R principal : "
        f"<b>{args.r_mode}</b>. "
        "T/F/TF/C/X : scores moyens entre canaux évaluables. "
        "R : cibles globales ou statistiques de paires.</p>",
        '<section><h2>Résumé par famille</h2><div class="scroll">',
        summary.round(4).to_html(escape=True),
        "</div></section>",
    ]

    for name, title, caption in charts:
        encoded = base64.b64encode(
            (out / (name + ".png")).read_bytes()
        ).decode("ascii")

        parts.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(caption)}</p>"
            f'<img alt="{html.escape(title)}" '
            f'src="data:image/png;base64,{encoded}">'
            "</section>"
        )

    table = all_results[[
        "Family",
        "Feature",
        "Best_Alpha",
        "Validation_R2",
        "Test_R2",
    ]].copy()

    parts += [
        '<section><h2>Tableau des 63 features</h2>'
        '<div class="scroll">',
        table.round(4).to_html(index=False, escape=True),
        "</div></section></main></html>",
    ]

    (out / "dashboard.html").write_text(
        "\n".join(parts),
        encoding="utf-8",
    )

    sources = {
        "r_mode": args.r_mode,
        "sources": {
            family: str(path.resolve())
            for family, path in paths.items()
        },
        "notice": NOTICE,
    }

    (out / "sources.json").write_text(
        json.dumps(sources, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n========== RÉSUMÉ ==========")
    print(summary.round(4).to_string())
    print(f"\nVisualisations : {out.resolve()}")
    print(
        f"Tableau de bord : "
        f"{(out / 'dashboard.html').resolve()}"
    )


if __name__ == "__main__":
    main()
