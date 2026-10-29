import numpy as np
import torch
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil :", device)

# Charger les modèles et les placer sur le même appareil
model = AutoModel.from_pretrained(
    "brain-bzh/reve-base",
    trust_remote_code=True,
    torch_dtype="auto"
).to(device)

pos_bank = AutoModel.from_pretrained(
    "brain-bzh/reve-positions",
    trust_remote_code=True
)

# Charger les données EEG
data = np.load("data/segments_labeled.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
y = data["y"]

# Ordre exact des 41 canaux
electrode_names = [
    "C2", "CP2", "P2", "Pz", "CPz", "Cz", "C1", "CP1",
    "P1", "P3", "PO7", "P5", "TP7", "CP5", "CP3", "C3",
    "PO8", "P6", "P4", "TP8", "CP6", "CP4", "C6", "C4",
    "FC6", "FC4", "FC2", "F6", "F4", "F2", "AF4", "AF3",
    "F5", "F3", "F1", "Fz", "FC5", "FC3", "FC1", "FCz", "C5"
]

# Calculer les positions sur le CPU, puis les transférer sur le GPU
positions = pos_bank(electrode_names)
positions = torch.as_tensor(positions, dtype=torch.float32)
positions = positions.unsqueeze(0).expand(X.shape[0], -1, -1).to(device)

# Extraire les embeddings
model.eval()

with torch.no_grad():
    output = model(X.to(device), positions)

embeddings = output.detach().cpu().numpy()

# Sauvegarder les embeddings et les labels
np.savez_compressed(
    "data/reve_embeddings.npz",
    embeddings=embeddings,
    y=y
)

print("Forme des EEG :", tuple(X.shape))
print("Forme des embeddings :", embeddings.shape)
print("Fichier sauvegardé : data/reve_embeddings.npz")