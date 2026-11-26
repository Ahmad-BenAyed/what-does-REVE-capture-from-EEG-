"""Ridge LOSO avec normalisation par sujet : REVE suit-il les variations intra-sujet ?

Trois versions du Ridge sur 308 cibles par défaut : six statistiques sur
les canaux pour 49 caractéristiques et 14 caractéristiques globales.

  A  brute               : aucune normalisation par sujet (reproduit le LOSO existant)
  B  cibles              : z-score des caractéristiques à l'intérieur de chaque sujet
  C  cibles+embeddings   : z-score des caractéristiques et des embeddings par sujet

Protocole : LOSO externe, LOSO interne sur les sujets d'entraînement pour choisir alpha
(par cible). Métriques par sujet de test : R² et r de Pearson. Contrôles : cibles
permutées à l'intérieur de chaque sujet, et décodage de l'identité du sujet.

Usage depuis la racine du dépôt :
  python task_1/loso_normalisation_sujet.py --check
  python task_1/loso_normalisation_sujet.py --out task_1/results_loso_norm_sujet
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .paths import Paths, output_problem
else:
    from paths import Paths, output_problem

STATS = {
    "mean": lambda a: a.mean(1),
    "std": lambda a: a.std(1),
    "median": lambda a: np.median(a, 1),
    "q25": lambda a: np.quantile(a, 0.25, axis=1),
    "q75": lambda a: np.quantile(a, 0.75, axis=1),
    "max": lambda a: a.max(1),
}
VERSIONS = {"A": "brute", "B": "cibles", "C": "cibles+embeddings"}
FAMILY_ORDER = ["T", "F", "TF", "C", "R", "X"]


# ---------------------------------------------------------------- données
def family_of(name):
    prefix = name.split("_")[0].rstrip("0123456789")
    return prefix if prefix in FAMILY_ORDER else "?"


def load_inputs(args):
    meta = pd.read_csv(args.metadata, dtype=str, keep_default_na=False)
    if "subject" not in meta:
        raise ValueError(f"{args.metadata} : colonne 'subject' absente")
    subjects = meta["subject"].str.strip().to_numpy(str)

    with np.load(args.features, allow_pickle=False) as f:
        per_channel = f["per_channel"].astype(np.float64)          # N × 41 × 49
        global_features = f["global_features"].astype(np.float64)  # N × 14
        ch_names = [str(s) for s in f["channel_feature_names"]]
        gl_names = [str(s) for s in f["global_feature_names"]]
        feat_labels = f["labels"] if "labels" in f.files else None

    targets, names, features = [], [], []
    for j, name in enumerate(ch_names):
        if args.par_canal:
            for channel in range(per_channel.shape[1]):
                targets.append(per_channel[:, channel, j])
                names.append((name, f"canal_{channel:02d}"))
        else:
            for stat, fn in STATS.items():
                targets.append(fn(per_channel[:, :, j]))
                names.append((name, stat))
    for j, name in enumerate(gl_names):
        targets.append(global_features[:, j])
        names.append((name, "global"))
    Y = np.stack(targets, 1)
    dims = pd.DataFrame(names, columns=["feature", "dimension"])
    dims["family"] = dims.feature.map(family_of)

    pooled_cache = args.out / "pooled_embeddings.npy" if args.out else None
    if args.pooled is not None:
        X = np.load(args.pooled)
        emb_labels = None
        if pooled_cache is not None and Path(args.pooled).resolve() != pooled_cache.resolve():
            np.save(pooled_cache, X)
    else:
        with np.load(args.embeddings, allow_pickle=False) as e:
            emb = e["embeddings"]                                   # N × 41 × 8 × 512
            emb_labels = e["y"] if "y" in e.files else None
            X = emb.reshape(len(emb), -1, emb.shape[-1]).mean(1, dtype=np.float64)
            del emb
        if pooled_cache is not None:
            np.save(pooled_cache, X.astype(np.float32))

    n = {"metadata": len(subjects), "features": len(Y), "embeddings": len(X)}
    if len(set(n.values())) != 1:
        raise ValueError(f"Nombres de lignes incohérents : {n}")
    if feat_labels is not None and emb_labels is not None and not np.array_equal(feat_labels, emb_labels):
        raise ValueError("Labels différents entre features_63.npz et reve_embeddings.npz : ordre des lignes ?")

    keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    return X[keep], Y[keep], subjects[keep], dims, int((~keep).sum())


def zscore_by_subject(A, subjects):
    """z-score de chaque colonne à l'intérieur de chaque sujet (stats du sujet lui-même).
    N'utilise aucun label de tâche ; colonnes constantes dans un sujet -> 0 et signalées."""
    out = np.zeros_like(A)
    constant = {}
    for s in np.unique(subjects):
        m = subjects == s
        mu, sd = A[m].mean(0), A[m].std(0)
        flat = sd <= 1e-12 * np.maximum(1, np.abs(mu))
        out[m] = np.where(flat, 0.0, (A[m] - mu) / np.where(flat, 1, sd))
        constant[s] = flat
    return out, constant


# ---------------------------------------------------------------- ridge
def ridge_predict(X_tr, Y_tr, X_te, alphas):
    """Ridge avec intercept pour toutes les cibles et tous les alphas (SVD)."""
    mx, sx = X_tr.mean(0), X_tr.std(0)
    sx[sx == 0] = 1
    Xc, Xt = (X_tr - mx) / sx, (X_te - mx) / sx
    my = Y_tr.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtY = U.T @ (Y_tr - my)
    XtV = Xt @ Vt.T
    return np.stack([XtV @ ((s / (s**2 + a))[:, None] * UtY) + my for a in alphas])  # A × n_te × T


def r2_and_r(y, p):
    """R² (référence : moyenne du sujet testé) et r de Pearson, colonne par colonne."""
    yc, pc = y - y.mean(0), p - p.mean(0)
    sst = (yc**2).sum(0)
    sse = ((y - p)**2).sum(0)
    denom = np.sqrt(sst * (pc**2).sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(sst > 0, 1 - sse / sst, np.nan)
        r = np.where(denom > 0, (yc * pc).sum(0) / denom, np.nan)
    return r2, r


def loso(X, Y, subjects, alphas, undefined):
    """LOSO externe + LOSO interne (choix d'alpha par cible sur le R² moyen intra-sujet)."""
    rows = []
    groups = np.unique(subjects)
    for test in groups:
        tr, te = subjects != test, subjects == test
        scores = []
        for val in groups[groups != test]:
            fit, held = tr & (subjects != val), subjects == val
            preds = ridge_predict(X[fit], Y[fit], X[held], alphas)
            scores.append(np.stack([r2_and_r(Y[held], p)[0] for p in preds]))
        validation = np.nanmean(np.stack(scores), 0)
        inner = np.where(np.isnan(validation), -np.inf, validation)
        best = inner.argmax(0)
        preds = ridge_predict(X[tr], Y[tr], X[te], alphas)
        p = preds[best, :, np.arange(Y.shape[1])].T
        r2, r = r2_and_r(Y[te], p)
        bad = undefined.get(test)
        if bad is not None:
            r2, r = np.where(bad, np.nan, r2), np.where(bad, np.nan, r)
        rows.append((test, r2, r, np.asarray(alphas)[best], validation[best, np.arange(Y.shape[1])]))
    return rows


def permute_within_subject(Y, subjects, rng):
    Yp = Y.copy()
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        Yp[idx] = Y[rng.permutation(idx)]
    return Yp


# ---------------------------------------------------------------- contrôle identité
def subject_identity(X, subjects, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, subjects, cv=cv)
    return balanced_accuracy_score(subjects, pred)


# ---------------------------------------------------------------- agrégation
def summarise(long):
    """Moyenne sur les dimensions d'une caractéristique dans chaque sujet, puis
    sur les caractéristiques d'une famille dans chaque sujet, puis entre sujets."""
    feat = (long.groupby(["version", "control", "subject", "family", "feature"])[["r2", "r"]]
            .mean().reset_index())
    real, perm = feat[feat.control == 0], feat[feat.control > 0]

    f_real = real.groupby(["version", "family", "feature"]).agg(
        r2_mean=("r2", "mean"), r2_median=("r2", "median"),
        r_mean=("r", "mean"), r_std=("r", "std"),
        n_subjects_r_positive=("r", lambda v: int((v > 0).sum())),
        n_subjects=("r", "count"))
    f_perm = (perm.groupby(["version", "family", "feature", "control"])["r"].mean()
              .groupby(["version", "family", "feature"]).agg(r_shuffle_mean="mean", r_shuffle_std="std"))
    features = f_real.join(f_perm).reset_index()

    fam_subj = feat.groupby(["version", "control", "subject", "family"])[["r2", "r"]].mean().reset_index()
    fr = fam_subj[fam_subj.control == 0].groupby(["version", "family"]).agg(
        r2_mean=("r2", "mean"), r2_median=("r2", "median"),
        r_mean=("r", "mean"), r_std_between_subjects=("r", "std"))
    fp = (fam_subj[fam_subj.control > 0].groupby(["version", "family", "control"])["r"].mean()
          .groupby(["version", "family"]).agg(r_shuffle_mean="mean", r_shuffle_max="max"))
    families = fr.join(fp).reset_index()
    families["family"] = pd.Categorical(families.family, FAMILY_ORDER, ordered=True)
    return features, families.sort_values(["version", "family"])


def figure(families, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.27
    x = np.arange(len(FAMILY_ORDER))
    for i, v in enumerate(VERSIONS):
        d = families[families.version == v].set_index("family").reindex(FAMILY_ORDER)
        lab = f"{v} · {VERSIONS[v]}"
        axes[0].bar(x + (i - 1) * width, d.r_mean, width, yerr=d.r_std_between_subjects, capsize=2, label=lab)
        axes[0].scatter(x + (i - 1) * width, d.r_shuffle_max, marker="_", color="k", s=80, zorder=3)
        axes[1].bar(x + (i - 1) * width, d.r2_median, width, label=lab)
    axes[0].set_title("r de Pearson par famille (trait noir = max des contrôles permutés)")
    axes[1].set_title("R² médian par famille")
    axes[1].set_yscale('symlog', linthresh=1)
    for ax in axes:
        ax.set_xticks(x, FAMILY_ORDER)
        ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def parse(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    defaults, _ = bootstrap.parse_known_args(argv)
    paths = Paths(defaults.config)
    data = paths.location("data_dir")
    local_data = Path(__file__).resolve().parents[1] / "data"
    if not data.is_dir() and local_data.is_dir():
        data = local_data
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--embeddings", type=Path, default=paths.input("embeddings", data))
    ap.add_argument("--features", type=Path, default=paths.input("features", data))
    ap.add_argument("--metadata", type=Path, default=paths.input("metadata", data))
    ap.add_argument("--pooled", type=Path, default=None,
                    help="pooled_embeddings.npy d'une exécution précédente (évite de relire 1,7 Go)")
    ap.add_argument("--out", type=Path, default=paths.location("results_dir"))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--versions", nargs="+", choices=list(VERSIONS), default=list(VERSIONS))
    ap.add_argument("--permutations", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true", help="vérifie les entrées et la sortie, sans calcul")
    ap.add_argument("--par-canal", action="store_true",
                    help="Experience optionnelle par indice de canal (sortie separee obligatoire)")
    args = ap.parse_args(argv)
    if args.par_canal and args.out.resolve() == paths.location("results_dir"):
        ap.error("--par-canal exige un --out distinct du dossier principal")
    if args.permutations < 0 or any(not np.isfinite(a) or a <= 0 for a in args.alphas):
        ap.error("Alphas strictement positifs et nombre de permutations positif ou nul requis.")
    if len(set(args.versions)) != len(args.versions):
        ap.error("Chaque version doit etre unique.")
    return args


def main(argv=None):
    args = parse(argv)
    needed = [args.metadata, args.features] + ([args.pooled] if args.pooled else [args.embeddings])
    missing = [str(p) for p in needed if not Path(p).is_file()]
    problem = output_problem(args.out)
    if missing or problem:
        for m in missing:
            print(f"[ERROR] Fichier introuvable : {m}")
        if str(args.features) in missing:
            print("Indiquez le fichier features_63.npz avec --features. "
                  "Ce fichier est produit par legacy/03_extract_63_features.py.")
        if problem:
            print(f"[ERROR] {problem}")
        return 1
    if args.check:
        print("[OK] Entrées présentes et dossier de sortie libre.")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(exist_ok=True)
    marker = args.out / ".running"
    with marker.open("x"):
        pass
    if any(p != marker for p in args.out.iterdir()):
        marker.unlink()
        print(f"[ERROR] Dossier de sortie devenu non vide : {args.out}")
        return 1
    X, Y, subjects, dims, dropped = load_inputs(args)
    print(f"{len(X)} essais, {len(np.unique(subjects))} sujets, {Y.shape[1]} cibles, {dropped} lignes non finies retirées")

    Yz, y_const = zscore_by_subject(Y, subjects)
    Xz, _ = zscore_by_subject(X, subjects)
    variants = {"A": (X, Y, {}), "B": (X, Yz, y_const), "C": (Xz, Yz, y_const)}
    rng = np.random.default_rng(args.seed)

    records = []
    validation_records = []
    for v in args.versions:
        Xv, Yv, undefined = variants[v]
        for control in range(args.permutations + 1):
            Yc = Yv if control == 0 else permute_within_subject(Yv, subjects, rng)
            for subject, r2, r, alpha, validation_r2 in loso(Xv, Yc, subjects, args.alphas, undefined):
                frame = dims.copy()
                frame["r2"], frame["r"], frame["alpha"] = r2, r, alpha
                frame["version"], frame["control"], frame["subject"] = v, control, subject
                records.append(frame)
                if control == 0:
                    validation_records.append(frame.drop(columns=["r2", "r", "control"]).assign(
                        validation_r2=validation_r2))
            print(f"  version {v} ({VERSIONS[v]}) · {'réel' if control == 0 else f'permutation {control}'} terminé")
    long = pd.concat(records, ignore_index=True)
    long.to_csv(args.out / "scores_par_sujet.csv", index=False)

    pd.concat(validation_records, ignore_index=True).to_csv(
        args.out / "validation_interne.csv", index=False)
    features, families = summarise(long)
    features.to_csv(args.out / "resume_caracteristiques.csv", index=False)
    families.to_csv(args.out / "resume_familles.csv", index=False)

    ident = pd.DataFrame([
        {"embeddings": "bruts", "balanced_accuracy": subject_identity(X, subjects, args.seed)},
        {"embeddings": "normalises_par_sujet", "balanced_accuracy": subject_identity(Xz, subjects, args.seed)},
    ])
    ident["chance"] = 1 / len(np.unique(subjects))
    ident.to_csv(args.out / "decodage_identite_sujet.csv", index=False)

    figure(families, args.out / "comparaison_familles.png")
    (args.out / "config.json").write_text(json.dumps({
        "date": datetime.now(timezone.utc).isoformat(),
        "embeddings": str(args.pooled or args.embeddings), "features": str(args.features),
        "metadata": str(args.metadata), "alphas": args.alphas, "versions": args.versions,
        "permutations": args.permutations, "seed": args.seed,
        "n_trials": int(len(X)), "n_subjects": int(len(np.unique(subjects))),
        "dropped_nonfinite_rows": dropped, "n_targets": int(Y.shape[1]),
        "target_mode": "canal" if args.par_canal else "statistiques",
        "n_embedding_dimensions": int(X.shape[1]),
        "targets": "49 features x indices de canaux + 14 globales" if args.par_canal else "49 features par canal x 6 statistiques sur les canaux + 14 globales",
        "pooling": "moyenne sur canaux et patchs -> 512",
        "outer_cv": "leave one subject out", "inner_cv": "leave one remaining subject out, alpha par cible",
        "subject_normalisation": "z-score par sujet avec les stats du sujet (aucun label de tâche)",
        "metrics": "R2 (référence = moyenne du sujet testé) et r de Pearson, par sujet",
        "aggregation": "dimensions -> caractéristique -> famille dans chaque sujet, puis entre sujets",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    marker.unlink()
    pd.set_option("display.width", 140)
    print("\nRésumé par famille :")
    print(families.round(3).to_string(index=False))
    print("\nDécodage de l'identité du sujet :")
    print(ident.round(3).to_string(index=False))
    print(f"\nRésultats : {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
