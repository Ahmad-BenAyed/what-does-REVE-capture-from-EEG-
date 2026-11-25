# Task 1 · Caractéristiques EEG décodables avec REVE

| Famille | Caractéristiques | Script dans `family/<famille>/code/` | Dossier sous `family/<famille>/resultats/` |
|---|---|---|---|
| T | Domaine temporel (10) | `Tresultar.py` | `results_temporal` |
| F | Spectre (16) | `Fresultat.py` | `results_spectral` |
| TF | Temps-fréquence (11) | `TFresultat.py` | `results_tf` |
| C | Complexité (7) | `Cresultat.py` | `results_c` |
| R | Relations entre canaux (14) | `Rresultat.py` | `results_r` |
| X | Relations entre fréquences (5) | `Xresultat.py` | `results_x` |

Les scripts utilisent `data/segments_labeled.npz` et `data/reve_embeddings.npz`. Ils produisent les caractéristiques, indices de partition et scores Ridge, ainsi que les prédictions sur le test et la configuration de l’expérience. Les partitions aléatoires par essai peuvent placer un même sujet dans plusieurs ensembles. Les R² de Task 1 mesurent le décodage dans ce protocole.

## Installation et lancement

Depuis la racine du dépôt, dans l’environnement Python de Task 1.

```bash
python -m pip install -r task_1/requirements.txt
python task_1/run_task_1.py --help
python task_1/run_task_1.py --family all --dry-run
python task_1/check_inputs.py --family all --out task_1/nouvelles-regressions
python task_1/run_task_1.py --family T --out task_1/essai-T
python task_1/run_task_1.py --family all --out task_1/nouvelles-regressions
```

Plusieurs familles se sélectionnent avec `--family T F`. Les options scientifiques se transmettent après `--`, par exemple `--family T --out task_1/essai-T -- --fs 200 --seed 42`. Pour une sélection de plusieurs familles, le runner applique ces options à chacune. Les arguments d’infrastructure se placent avant `--`.

Les scripts individuels acceptent `--segments`, `--embeddings`, `--out`, `--config` et les options propres à chaque famille. Ils s’exécutent également depuis leur dossier `code/` avec `python NOM_DU_SCRIPT.py`.

```bash
python task_1/family/T/code/Tresultar.py --help
python task_1/family/F/code/Fresultat.py --help
python task_1/family/TF/code/TFresultat.py --help
python task_1/family/C/code/Cresultat.py --help
python task_1/family/R/code/Rresultat.py --help
python task_1/family/X/code/Xresultat.py --help
```

## Visualisation et chemins

```bash
python task_1/check_inputs.py --visualize --out task_1/figures-nouvelles
python task_1/run_task_1.py --visualize --out task_1/figures-nouvelles
python task_1/run_task_1.py --visualize --root task_1/nouvelles-regressions --out task_1/figures-nouvelles-regressions
```

La visualisation lit les six CSV principaux, ajoute les détails par canal disponibles et compare les représentations R si les deux jeux de scores sont présents. Elle écrit des figures PNG/PDF, des CSV, un JSON de provenance et `dashboard.html`. `--r-mode summary` choisit la variante R historique. `--root` attend un arbre au format `<famille>/resultats/<dossier>`. Avec une seule famille, choisissez directement son dossier final sous cet arbre pour la combiner à cinq autres sorties disposées de la même façon.

`config.json` gère les chemins d’infrastructure. Les chemins de la configuration se résolvent depuis le dossier du JSON, et les chemins explicites sur la ligne de commande depuis le dossier courant. Le runner accepte un JSON alternatif avec `--config`. Les valeurs intégrées servent de repli lorsque des clés manquent.

Un dossier de sortie non vide entraîne un refus, y compris en lancement direct. Choisissez un nouveau `--out` si vous utilisez les résultats historiques comme source. `--check` vérifie les fichiers, les dépendances et la destination sans ouvrir les grands tableaux. `--dry-run` affiche le plan et ne lance aucun processus enfant. Chaque lancement orchestré écrit un nouveau fichier sous `task_1/logs/`.

En cas de données absentes, remettez les NPZ locaux dans `data/` ou fournissez les deux chemins explicites. Si une bibliothèque manque, installez les dépendances de Task 1 dans l’environnement du lanceur.

```bash
python -m unittest discover -s task_1/tests -v
```
