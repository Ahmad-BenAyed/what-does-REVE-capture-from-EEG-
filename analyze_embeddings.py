
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score

os.makedirs("results", exist_ok=True)

# Charger les embeddings et les labels
data = np.load("data/reve_embeddings.npz")
E = data["embeddings"]
y = data["y"]

print("Embeddings :", E.shape)
print("Labels :", y.shape)
print("NaN :", np.isnan(E).sum())
print("Inf :", np.isinf(E).sum())

# Moyenne sur les électrodes et les positions temporelles
# E : (segments, électrodes, temps, dimensions latentes)
X = E.mean(axis=(1, 2))

print("Représentation par segment :", X.shape)

# Associer les noms aux labels
class_names = {
    0: "Imagerie motrice",
    1: "Soustraction",
    2: "Génération de mots"
}
print("Effectifs :", {
    class_names[int(k)]: int(v)
    for k, v in zip(*np.unique(y, return_counts=True))
})

# PCA en 2 dimensions pour visualiser les classes
X_scaled = StandardScaler().fit_transform(X)
X_2d = PCA(n_components=2).fit_transform(X_scaled)

plt.figure(figsize=(9, 7))
for label, name in class_names.items():
    mask = y == label
    plt.scatter(
        X_2d[mask, 0], X_2d[mask, 1],
        s=12, alpha=0.55, label=name
    )

plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("PCA des embeddings REVE")
plt.legend()
plt.tight_layout()
plt.savefig("results/pca_embeddings.png", dpi=180)
plt.close()

# Évaluation avec validation croisée stratifiée
clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(
    clf, X, y,
    cv=cv,
    scoring={
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro"
    },
    n_jobs=1
)

print("\n=== Résultats exploratoires ===")
print(
    "Balanced accuracy : "
    f"{scores['test_balanced_accuracy'].mean():.3f} "
    f"+/- {scores['test_balanced_accuracy'].std():.3f}"
)
print(
    "F1 macro : "
    f"{scores['test_f1_macro'].mean():.3f} "
    f"+/- {scores['test_f1_macro'].std():.3f}"
)

print("\nGraphique enregistré : results/pca_embeddings.png")
