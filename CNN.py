
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------
SEED = 42
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Appareil :", device)

# --------------------------------------------------
# 2. Chargement des embeddings REVE
# --------------------------------------------------
data = np.load("data/reve_embeddings.npz")
X = data["embeddings"].astype(np.float32)
y = data["y"].astype(np.int64)

print("Embeddings :", X.shape)
print("Labels :", y.shape)
print("Classes et effectifs :", np.unique(y, return_counts=True))

# X : (essais, canaux, temps, caractéristiques)
assert X.ndim == 4, "Les embeddings doivent avoir 4 dimensions."
assert len(X) == len(y), "Le nombre de labels doit correspondre aux essais."

# Vérifier les trois classes
classes = np.unique(y)
if len(classes) != 3:
    raise ValueError(
        f"Il faut 3 classes dans y, mais on en trouve {classes}. "
        "Vérifie les labels de segments_labeled.npz."
    )

# --------------------------------------------------
# 3. Séparation entraînement / test
# --------------------------------------------------
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices,
    test_size=0.25,
    random_state=SEED,
    stratify=y,
)

# Standardisation sans fuite des données de test.
# Une moyenne/écart-type par dimension d'embedding,
# calculés uniquement sur les essais d'entraînement.
scaler = StandardScaler()
X_train = X[train_idx]
X_test = X[test_idx]

scaler.fit(X_train.reshape(-1, X.shape[-1]))

X_train = scaler.transform(
    X_train.reshape(-1, X.shape[-1])
).reshape(X_train.shape)

X_test = scaler.transform(
    X_test.reshape(-1, X.shape[-1])
).reshape(X_test.shape)

# CNN PyTorch : (batch, canaux d'entrée, hauteur, largeur)
# Ici : hauteur = électrodes, largeur = positions temporelles.
X_train = torch.tensor(X_train).unsqueeze(1)
X_test = torch.tensor(X_test).unsqueeze(1)

y_train = torch.tensor(y[train_idx], dtype=torch.long)
y_test = torch.tensor(y[test_idx], dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# --------------------------------------------------
# 4. Définition du CNN
# --------------------------------------------------
class EEGEmbeddingCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = EEGEmbeddingCNN(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# --------------------------------------------------
# 5. Entraînement
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(yb)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- loss train: {total_loss / len(train_idx):.4f}"
    )

# --------------------------------------------------
# 6. Évaluation sur le test
# --------------------------------------------------
model.eval()
with torch.no_grad():
    logits = model(X_test.to(device))
    predictions = logits.argmax(dim=1).cpu().numpy()

y_true = y_test.numpy()

print("\nAccuracy :", accuracy_score(y_true, predictions))
print(
    "Balanced accuracy :",
    balanced_accuracy_score(y_true, predictions),
)
print("\nRapport de classification :")
print(classification_report(y_true, predictions, zero_division=0))
print("Matrice de confusion :")
print(confusion_matrix(y_true, predictions))

# Sauvegarder le modèle entraîné
torch.save(model.state_dict(), "data/cnn_classifier.pt")
print("\nModèle sauvegardé : data/cnn_classifier.pt")
