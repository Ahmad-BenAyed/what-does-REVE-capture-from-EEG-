# Que capture REVE dans les signaux EEG ?

Ce projet étudie ce que les représentations du modèle EEG préentraîné REVE (`brain-bzh/reve-base`) encodent. Les essais EEG de `nm000290` correspondent à trois tâches mentales, l’imagerie motrice (`mi`), le calcul mental (`sub`) et la génération de mots (`word`).

- [Task 1](task_1/README.md) estime dans quelle mesure les embeddings permettent de prédire des caractéristiques EEG classiques par régression Ridge.
- [Task 2](task_2/README.md) classe les tâches mentales à partir des embeddings avec un MLP ou un CNN en LOSO, puis étudie l’importance des caractéristiques EEG par effacement LEACE sur le MLP.

Chaque tâche a ses scripts, ses dépendances, sa configuration et ses résultats. Les deux tâches peuvent lire les mêmes données sources locales.

```text
.
├── prepare_data/       Préparation des essais et extraction des embeddings REVE
├── task_1/             Régressions, six familles et visualisations
├── task_2/             Classification, cibles EEG propres et résultats LEACE
├── legacy/             Scripts de l’ancien pipeline
├── data/               Entrées locales des deux tâches (ignoré par Git)
└── raw_data/           Enregistrements locaux (ignoré par Git)
```

## Données locales

| Fichier dans `data/` | Contenu |
|---|---|
| `segments_labeled.npz` | Signaux par essai et canal (`X`), labels (`y`) |
| `segments_metadata.csv` | Métadonnées par essai dans le même ordre, avec le sujet |
| `features_63.npz` | Cibles EEG de Task 1, produites par `legacy/03_extract_63_features.py` |
| `reve_embeddings.npz` | Embeddings REVE par essai, canal et fenêtre, avec les labels (`y`) |

Les données EEG et les embeddings sont volumineux et restent locaux. Les contrôles structurels et les tests synthétiques fonctionnent sur un dépôt sans ces fichiers.

## Préparation des données

Depuis la racine du dépôt, avec les EDF et TSV BIDS en place sous `raw_data/nm000290/` et un environnement dédié à la préparation.

```bash
python -m pip install -r prepare_data/requirements.txt
python prepare_data/01_prepare_all_data.py
python prepare_data/02_embeddings_all_data.py
```

Le premier script rééchantillonne à 200 Hz, applique un passe-bande à 0,5–99,5 Hz et normalise par canal et par enregistrement avant de découper les essais. Il garde des essais de forme identique. Le second utilise les positions de 41 électrodes et extrait les embeddings REVE par lots de 16. Il utilise CUDA si disponible et peut récupérer les modèles Hugging Face si le cache local est absent. Les scripts refusent d’écraser leurs fichiers de sortie.

Le [dossier historique](legacy/README.md) conserve les scripts `03` et `04`. Consultez les guides des deux tâches pour exécuter les expériences actives.
