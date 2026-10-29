import numpy as np

data = np.load("data/segments_labeled.npz")
X = data["X"]
y = data["y"]

print("Shape X :", X.shape)
print("Shape y :", y.shape)
print("Type :", X.dtype)
print("Valeurs NaN :", np.isnan(X).sum())
print("Valeurs infinies :", np.isinf(X).sum())
print("Minimum :", X.min())
print("Maximum :", X.max())
print("Moyenne :", X.mean())
print("Écart-type :", X.std())
print("Durée d'un segment :", X.shape[2] / 200, "secondes")
