
import numpy as np
import torch
from transformers import AutoModel

# Configuration
BATCH_SIZE = 16
INPUT_FILE = "data/segments_labeled.npz"
OUTPUT_FILE = "data/reve_embeddings.npz"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil :", device)

# Charger REVE
model = AutoModel.from_pretrained(
    "brain-bzh/reve-base",
    trust_remote_code=True,
    torch_dtype="auto"
).to(device)
model.eval()

# Charger les positions des électrodes
pos_bank = AutoModel.from_pretrained(
    "brain-bzh/reve-positions",
    trust_remote_code=True
)

# Charger toutes les données
data = np.load(INPUT_FILE)
X = data["X"]
y = data["y"]

print("Forme des EEG :", X.shape)
print("Nombre de segments :", len(X))

# Ordre exact des 41 canaux
electrode_names = [
    "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1",
    "P1", "P3", "PO7", "P5", "TP7", "CP5", "CP3", "C3",
    "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6", "C4",
    "FC6", "FC4", "FC2", "F6", "F4", "F2", "AF4", "AF3",
    "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1", "FCz", "C5"
]

# Calculer les positions une seule fois
positions = pos_bank(electrode_names)
positions = torch.as_tensor(positions, dtype=torch.float32)

# Ajouter la dimension batch
if positions.ndim == 2:
    positions = positions.unsqueeze(0)

# Extraire les embeddings par lots
all_embeddings = []

with torch.inference_mode():
    for start in range(0, len(X), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X))

        batch_X = torch.as_tensor(
            X[start:end], dtype=torch.float32, device=device
        )

        batch_positions = positions.expand(
            batch_X.shape[0], -1, -1
        ).to(device)

        output = model(batch_X, batch_positions)

        # Garder les résultats en mémoire CPU
        batch_embeddings = output.detach().float().cpu().numpy()
        all_embeddings.append(batch_embeddings)

        print(
            f"Progression : {end}/{len(X)} segments",
            flush=True
        )

        del batch_X, batch_positions, output

# Regrouper tous les embeddings
embeddings = np.concatenate(all_embeddings, axis=0)

# Sauvegarder les embeddings et les labels
np.savez_compressed(
    OUTPUT_FILE,
    embeddings=embeddings,
    y=y
)

print("\n=== RÉSULTATS ===")
print("Forme des EEG :", X.shape)
print("Forme des embeddings :", embeddings.shape)
print("Forme des labels :", y.shape)
print("Fichier sauvegardé :", OUTPUT_FILE)
