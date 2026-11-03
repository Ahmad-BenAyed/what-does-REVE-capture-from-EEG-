
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ============================================================
# CONFIGURATION
# ============================================================

EMB_FILE = "reve_embeddings.npz"
FEATURE_FILE = "features_63.npz"

ALPHAS = [0.1, 1.0, 10.0, 100.0]
SEED = 42

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n========== LOADING DATA ==========")

emb_data = np.load(EMB_FILE, allow_pickle=True)
feat_data = np.load(FEATURE_FILE, allow_pickle=True)

print("Embedding keys:", emb_data.files)
print("Feature keys:", feat_data.files)

E = np.asarray(emb_data["embeddings"], dtype=np.float32)

# 49 features per channel: (N, C, 49)
P = np.asarray(feat_data["per_channel"], dtype=np.float32)

# 14 global features: (N, 14)
G = np.asarray(feat_data["global_features"], dtype=np.float32)

channel_names = feat_data["channel_feature_names"].astype(str).tolist()
global_names = feat_data["global_feature_names"].astype(str).tolist()

print("\nEmbedding shape:", E.shape)
print("Per-channel features:", P.shape)
print("Global features:", G.shape)

# ============================================================
# 2. PREPARE REVE EMBEDDINGS
# ============================================================

# Same pooling as previous experiment:
# (N, 41, 8, 512) -> (N, 512)

if E.ndim > 2:
    E = E.mean(axis=tuple(range(1, E.ndim - 1)))

if E.ndim != 2:
    raise ValueError(f"Unexpected embedding shape: {E.shape}")

N, D = E.shape

if P.ndim != 3 or G.ndim != 2:
    raise ValueError("Unexpected feature dimensions.")

if P.shape[0] != N or G.shape[0] != N:
    raise ValueError("Embedding and feature sample counts differ.")

C = P.shape[1]
n_per_channel = P.shape[2]
n_global = G.shape[1]

if len(channel_names) != n_per_channel:
    raise ValueError("Channel feature names do not match data.")

if len(global_names) != n_global:
    raise ValueError("Global feature names do not match data.")

print("\n========== DATA INFORMATION ==========")
print("Segments:", N)
print("Embedding dimension:", D)
print("EEG channels:", C)
print("Per-channel features:", n_per_channel)
print("Global features:", n_global)
print("Total features:", n_per_channel + n_global)

# ============================================================
# 3. REMOVE INVALID SEGMENTS
# ============================================================

valid = (
    np.isfinite(E).all(axis=1)
    & np.isfinite(P).all(axis=(1, 2))
    & np.isfinite(G).all(axis=1)
)

E = E[valid]
P = P[valid]
G = G[valid]

print("\nValid segments:", len(E))

# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

indices = np.arange(len(E))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED
)

print("\n========== DATA SPLIT ==========")
print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

# ============================================================
# 5. STANDARDIZE EMBEDDINGS
# ============================================================

x_scaler = StandardScaler()

X_train = x_scaler.fit_transform(E[train_idx])
X_val = x_scaler.transform(E[val_idx])
X_test = x_scaler.transform(E[test_idx])

X_trainval = np.concatenate([X_train, X_val])

# ============================================================
# 6. RIDGE PROBING
# ============================================================

results = []
channel_results = []

def evaluate_ridge(feature_name, Y, family, channel_names=None):

    """
    Y:
      Per-channel feature: (N, C)
      Global feature:      (N,)
    """

    global train_idx, val_idx, test_idx
    global X_train, X_val, X_test, X_trainval

    Y = np.asarray(Y, dtype=np.float32)

    if Y.ndim == 1:
        Y = Y[:, None]

    # Standardize target using training data only
    y_scaler = StandardScaler()

    Y_train = y_scaler.fit_transform(Y[train_idx])
    Y_val = y_scaler.transform(Y[val_idx])
    Y_test = y_scaler.transform(Y[test_idx])

    best_alpha = None
    best_val_r2 = -np.inf

    # Hyperparameter selection
    for alpha in ALPHAS:

        model = Ridge(alpha=alpha)
        model.fit(X_train, Y_train)

        pred_val = model.predict(X_val)

        val_r2 = r2_score(
            Y_val,
            pred_val,
            multioutput="uniform_average"
        )

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_alpha = alpha

    # Final training on train + validation
    Y_trainval = np.concatenate([Y_train, Y_val])

    final_scaler = StandardScaler()
    Y_trainval_original = np.concatenate([
        Y[train_idx],
        Y[val_idx]
    ])

    Y_trainval_scaled = final_scaler.fit_transform(
        Y_trainval_original
    )

    final_model = Ridge(alpha=best_alpha)

    final_model.fit(
        X_trainval,
        Y_trainval_scaled
    )

    # Independent test evaluation
    pred_test = final_model.predict(X_test)
    pred_test = final_scaler.inverse_transform(pred_test)

    channel_r2 = r2_score(
        Y[test_idx],
        pred_test,
        multioutput="raw_values"
    )

    mean_test_r2 = float(np.mean(channel_r2))

    results.append({
        "Family": family,
        "Feature": feature_name,
        "Best_Alpha": best_alpha,
        "Validation_R2": best_val_r2,
        "Test_R2": mean_test_r2
    })

    # Save individual channel results
    if channel_names is not None:
        for ch, score in zip(channel_names, channel_r2):
            channel_results.append({
                "Feature": feature_name,
                "Channel": ch,
                "Test_R2": score
            })

    print(
        f"{family:4s} | "
        f"{feature_name:12s} | "
        f"Alpha={best_alpha:6.1f} | "
        f"Val_R2={best_val_r2:.4f} | "
        f"Test_R2={mean_test_r2:.4f}"
    )


print("\n========== PER-CHANNEL FEATURES ==========")

# T, F, TF, C and X: 49 features
# Each model predicts the feature across all EEG channels.

for j, feature_name in enumerate(channel_names):

    Y = P[:, :, j]  # (N, 41)

    family = feature_name.split("_")[0]

    channel_labels = [
        f"Channel_{ch+1}" for ch in range(C)
    ]

    evaluate_ridge(
        feature_name=feature_name,
        Y=Y,
        family=family,
        channel_names=channel_labels
    )


print("\n========== GLOBAL FEATURES ==========")

# R: 14 global features
# Each model predicts one value per segment.

for j, feature_name in enumerate(global_names):

    Y = G[:, j]  # (N,)

    evaluate_ridge(
        feature_name=feature_name,
        Y=Y,
        family="R"
    )

# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "reve_63_ridge_results.csv",
    index=False
)

channel_df = pd.DataFrame(channel_results)

channel_df.to_csv(
    "reve_63_channel_results.csv",
    index=False
)

# ============================================================
# 8. FAMILY SUMMARY
# ============================================================

family_summary = (
    results_df.groupby("Family")
    .agg(
        Number_of_Features=("Feature", "count"),
        Mean_Test_R2=("Test_R2", "mean"),
        Std_Test_R2=("Test_R2", "std"),
        Min_Test_R2=("Test_R2", "min"),
        Max_Test_R2=("Test_R2", "max")
    )
    .reset_index()
)

family_summary.to_csv(
    "reve_63_family_summary.csv",
    index=False
)

# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n========== RESULTS BY FEATURE ==========")
print(
    results_df.sort_values(
        "Test_R2",
        ascending=False
    ).to_string(index=False)
)

print("\n========== RESULTS BY FAMILY ==========")
print(family_summary.to_string(index=False))

print("\n========== SAVED FILES ==========")
print("1. reve_63_ridge_results.csv")
print("2. reve_63_channel_results.csv")
print("3. reve_63_family_summary.csv")
