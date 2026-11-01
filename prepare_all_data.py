
from pathlib import Path
from importlib import import_module
import pandas as pd
import numpy as np

try:
    mne = import_module("mne")
except ModuleNotFoundError as e:
    if e.name != "mne":
        raise
    raise RuntimeError(
        "MNE is required to read EDF files. Install it with: python -m pip install mne"
    ) from e

# Dossiers
DATA_DIR = Path("raw_data/nm000290")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Labels
label_map = {"mi": 0, "sub": 1, "word": 2}

X = []
y = []
metadata = []

edf_files = sorted(DATA_DIR.rglob("*_eeg.edf"))

print(f"Nombre de fichiers EDF trouvés : {len(edf_files)}")

for i, edf_file in enumerate(edf_files, 1):
    events_file = edf_file.with_name(
        edf_file.name.replace("_eeg.edf", "_events.tsv")
    )

    if not events_file.exists():
        print(f"[IGNORÉ] Événements absents : {edf_file.name}")
        continue

    try:
        print(f"[{i}/{len(edf_files)}] Traitement : {edf_file.name}")

        # Lire le signal et les événements
        raw = mne.io.read_raw_edf(
            edf_file, preload=True, verbose=False
        )
        events = pd.read_csv(events_file, sep="\t")

        # Rééchantillonnage à 200 Hz
        raw.resample(200, verbose=False)
        sfreq = raw.info["sfreq"]

        # Filtrage passe-bande
        raw.filter(l_freq=0.5, h_freq=99.5, verbose=False)

        signal = raw.get_data().astype(np.float32)

        # Normalisation Z-score par canal et par fichier
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        signal = (signal - mean) / std
        signal = np.clip(signal, -15.0, 15.0)

        # Extraire les essais correspondant aux 3 classes
        for _, event in events.iterrows():
            trial_type = str(event["trial_type"]).strip().lower()

            if trial_type not in label_map:
                continue

            start = int(float(event["onset"]) * sfreq)
            duration = float(event["duration"])
            end = int((float(event["onset"]) + duration) * sfreq)

            # Ignorer les événements invalides ou hors du signal
            if start < 0 or end > signal.shape[1] or end <= start:
                continue

            segment = signal[:, start:end]

            # Garder uniquement les segments de même longueur
            # pour pouvoir les regrouper dans un tableau NumPy
            if len(X) > 0 and segment.shape != X[0].shape:
                continue

            X.append(segment)
            y.append(label_map[trial_type])
            metadata.append({
                "subject": edf_file.parts[-4]
                    if len(edf_file.parts) >= 4 else "unknown",
                "file": edf_file.name,
                "trial_type": trial_type,
                "onset": float(event["onset"]),
                "duration": duration,
            })

        raw.close()

    except Exception as e:
        print(f"[ERREUR] {edf_file.name}: {e}")

if not X:
    raise RuntimeError(
        "Aucun segment valide extrait. Vérifie les labels et les événements."
    )

X = np.stack(X).astype(np.float32)
y = np.asarray(y, dtype=np.int64)

metadata_df = pd.DataFrame(metadata)

np.savez_compressed(
    OUTPUT_DIR / "segments_labeled.npz",
    X=X,
    y=y,
)

metadata_df.to_csv(
    OUTPUT_DIR / "segments_metadata.csv", index=False
)

print("\n=== RÉSULTATS ===")
print("Forme des segments :", X.shape)
print("Forme des labels :", y.shape)
print("Nombre d'essais par classe :", {
    label: int(np.sum(y == class_id))
    for label, class_id in label_map.items()
})
print("Sujets représentés :", metadata_df["subject"].nunique())
print("Métadonnées :", OUTPUT_DIR / "segments_metadata.csv")
print("Segments :", OUTPUT_DIR / "segments_labeled.npz")

