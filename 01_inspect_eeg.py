from pathlib import Path
import mne
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
edf_file = DATA_DIR / "sub-P01_ses-1_task-AcquisitionGraz_run-1_eeg.edf"
events_file = DATA_DIR / "sub-P01_ses-1_task-AcquisitionGraz_run-1_events.tsv"

# Lire le signal EEG et les événements
raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
events = pd.read_csv(events_file, sep="\t")
raw.resample(200)

# Correspondance entre les événements et les labels
label_map = {"mi": 0, "sub": 1, "word": 2}

X = []
y = []

sfreq = raw.info["sfreq"]
# Filtrage passe-bande
raw.filter(l_freq=0.5, h_freq=99.5, verbose=False)

# Récupérer le signal filtré
signal = raw.get_data().astype(np.float32)

# Normalisation Z-score par canal sur toute la session
mean = signal.mean(axis=1, keepdims=True)
std = signal.std(axis=1, keepdims=True)
std[std == 0] = 1.0

signal = (signal - mean) / std

# Écrêtage à ±15 écarts-types
signal = np.clip(signal, -15.0, 15.0)

for _, event in events.iterrows():
    trial_type = event["trial_type"]

    if trial_type not in label_map:
        continue

    start = int(event["onset"] * sfreq)
    duration = float(event["duration"])
    end = int((event["onset"] + duration) * sfreq)

    

    segment = signal[:, start:end]
    X.append(segment)
    y.append(label_map[trial_type])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# Sauvegarder les segments et leurs labels
np.savez_compressed(
    DATA_DIR / "segments_labeled.npz",
    X=X,
    y=y,
)

print("Forme des segments :", X.shape)
print("Forme des labels :", y.shape)
print("Labels : 0=mi, 1=sub, 2=word")
print("Nombre d'essais par classe :", {
    label: int(np.sum(y == label)) for label in np.unique(y)
})