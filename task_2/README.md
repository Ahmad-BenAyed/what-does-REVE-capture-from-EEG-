# Task 2 · Classification LOSO et effacement LEACE

Les embeddings REVE servent à classer les trois tâches mentales. Le CNN conserve la structure des électrodes et des fenêtres. Le MLP moyenne ces deux dimensions, standardise les vecteurs à partir du train et comporte une couche cachée de 64 unités par défaut. Le [guide scientifique du MLP](classification/README_MLP.md) décrit l’architecture, les précautions d’interprétation et les sorties par pli.

La validation externe tient un sujet à l’écart par pli. Le nombre d’époques est choisi par la médiane des meilleures époques de trois plis internes GroupKFold par sujet, puis le modèle est réentraîné sur les sujets d’entraînement externes avant le test. Le MLP inclut tous les sujets par défaut. Le CNN exclut `sub-P02` par défaut, conformément à sa configuration existante.

## Entraînement et évaluation

Depuis la racine du dépôt avec l’environnement Python de Task 2.

```bash
python -m pip install -r task_2/requirements.txt
python task_2/check_inputs.py --step train_mlp --out task_2/classification/mlp-nouveau
python task_2/run_task_2.py --step train_mlp --out task_2/classification/mlp-nouveau
python task_2/run_task_2.py --step train_cnn --out task_2/classification/cnn-nouveau
```

Les commandes directes utilisent le contexte de package.

```bash
python -m task_2.classification.train_mlp_loso --out task_2/classification/mlp-nouveau
python -m task_2.classification.train_cnn_loso --out task_2/classification/cnn-nouveau
python -m task_2.classification.evaluate_frozen_mlp --checkpoint task_2/classification/results_mlp_tous_sujets/fold_00/checkpoint.pt --out task_2/classification/mlp-gele-fold00
```

L’évaluation gelée accepte `--embeddings` et `--metadata` et requiert un checkpoint et un dossier de sortie. Elle utilise les poids, la normalisation, les étiquettes et les indices de test enregistrés. Pour un CNN, utilisez `python -m task_2.classification.evaluate_frozen_cnn` et le checkpoint CNN du pli voulu. Les deux points d’entrée utilisent l’évaluateur existant, qui lit le type de modèle dans le checkpoint. Le runner offre `evaluate_mlp` et `evaluate_cnn` avec ces mêmes chemins.

Les sorties d’entraînement incluent les métriques, prédictions, partitions LOSO, contrôles qualité, configuration et checkpoints par pli. Un dossier non vide est protégé. Pour reprendre une exécution interrompue, passez `--resume` avec les mêmes options et le même `--out`. Le script d’entraînement contrôle la compatibilité avant de reprendre.

## LEACE et visualisation

LEACE utilise les embeddings et métadonnées de `data/`, les dix checkpoints MLP `fold_00` à `fold_09`, leurs `predictions.csv` et le `config.json` de l’expérience MLP. Il lit six archives `test_predictions.npz` sous `task_2/family/<famille>/resultats/<dossier>/`.

| Famille | Dossier |
|---|---|
| T | `results_temporal` |
| F | `results_spectral` |
| TF | `results_tf` |
| C | `results_c` |
| R | `results_r` |
| X | `results_x` |

Ces six archives sont présentes. Les scripts ayant généré ces cibles ne figurent pas dans Task 2. Leur régénération n’est donc pas reproductible depuis ce dépôt seul. Les calculs de Task 1 suivent leur propre protocole et ses sorties sont exclues des entrées de LEACE.

```bash
python task_2/run_task_2.py --step leace --dry-run
python task_2/run_task_2.py --step leace --check --out task_2/classification/leace-nouveau
python task_2/run_task_2.py --step leace --out task_2/classification/leace-nouveau
python -m task_2.classification.leace_mlp --help
python task_2/run_task_2.py --step visualize --out task_2/classification/figures-leace-nouvelles
python -m task_2.classification.visualiser_leace --results task_2/classification/leace-nouveau --out task_2/classification/leace-nouveau/visualisations
```

LEACE vérifie l’alignement, les empreintes, les exclusions LOSO et la reproduction des probabilités du MLP. Il écrit notamment `by_subject.csv`, `summary.csv` et les facteurs d’effacement par sujet et caractéristique. Le cache temporaire des embeddings peut demander un espace disque comparable à la taille décompressée du membre NPZ. La visualisation lit `by_subject.csv` et produit des PNG, un PDF, un CSV et une page HTML.

## Configuration, contrôles et logs

`task_2/config.json` fournit les chemins par défaut. Les chemins du JSON sont relatifs à son dossier. Les options explicites sont relatives au dossier courant. `--config` sélectionne un JSON alternatif dans le runner. Les options d’infrastructure (dont `--data`, `--embeddings`, `--metadata`, `--models`, `--family`, `--checkpoint`, `--results`, `--out` selon l’étape) se placent avant `--`. Placez les options d’expérience après `--`, par exemple `--step leace --out task_2/classification/leace-nouveau -- --controls 10`.

`--check` contrôle l’étape demandée. Pour l’entraînement et LEACE, il lit les en-têtes NPZ et compte les lignes du CSV de métadonnées. Pour LEACE, il vérifie aussi la structure des cibles et la présence des dix paires checkpoint/prédictions. `--deep` avec `--check` lit les tableaux de labels et, pour LEACE, calcule l’empreinte des métadonnées. Le coût dépend du nombre de lignes et de la taille des métadonnées. Le contrôle des poids et la reproduction des prédictions s’effectuent pendant l’expérience. Un code non nul indique un contrôle en échec, y compris pour une destination déjà occupée.

`--dry-run` affiche la commande sans lancer l’étape. Les exécutions orchestrées écrivent chacune un nouveau fichier sous `task_2/logs/`. Les sorties détaillées des programmes enfants, qui peuvent contenir les identifiants des sujets, restent dans le terminal. Utilisez un nouveau `--out` pour conserver les résultats de référence.

```bash
python -m unittest discover -s task_2/tests -v
```

## CNN LOSO tous sujets

Le lancement historique `--step train_cnn` exclut toujours `sub-P02` et conserve
sa destination `results_cnn_sans_P02`. La nouvelle option `--all-subjects`
inclut tous les sujets éligibles et utilise `results_cnn_tous_sujets`.
Le script direct accepte aussi cette option. La cohorte dépend de la
présence de lignes finies pour chaque sujet. Le CNN et le MLP partagent le
contrôle des lignes, les métriques, les graines et le protocole LOSO.
Le CNN conserve sa largeur 32, contre 64 pour le MLP. Les deux conservent
le dropout 0,25, AdamW avec LR et décroissance du poids à 0,001, des lots
de 16, trois plis internes par sujet et au plus 20 époques avec patience 5.
Les statistiques de normalisation sont ajustées uniquement sur les lignes
d'entraînement de chaque ajustement, sur les tokens pour le CNN et sur les
vecteurs moyennés pour le MLP.

```bash
python task_2/run_task_2.py --step train_cnn --all-subjects --check
python task_2/run_task_2.py --step train_cnn --all-subjects -- --device cuda
python task_2/run_task_2.py --step train_cnn --all-subjects --resume -- --device cuda
```

Chaque pli `fold_XX` sauvegarde `checkpoint.pt`, `predictions.csv`,
`confusion_matrix.csv`, `metrics.json`, `train_history.csv`,
`inner_history.csv` et `inner_splits.csv`. La racine de l'expérience contient
`config.json`, `fold_metrics.csv`, `metrics_summary.csv`, `predictions.csv`,
`outer_splits.csv`, `rows_qc.csv` et `class_counts_by_subject.csv`.

### Relecture gelée d'un pli CNN

```bash
python task_2/run_task_2.py --step evaluate_cnn --checkpoint task_2/classification/results_cnn_tous_sujets/fold_00/checkpoint.pt --reference task_2/classification/results_cnn_tous_sujets/fold_00/predictions.csv --out task_2/classification/cnn-relecture-fold00
```

`--reference` est facultatif et ajoute les métriques originales, les écarts et
l'écart maximal entre probabilités. Pour une intervention, passez
`--embeddings` avec un NPZ complet dans le même ordre que l'original, avec
les mêmes labels et la même forme N × canaux × fenêtres × dimensions.
Les tokens originaux ou transformés entrent dans le CNN avant la
normalisation sauvegardée.

### LEACE CNN par projection affine commune

```bash
python task_2/run_task_2.py --step leace_cnn --dry-run
python task_2/run_task_2.py --step leace_cnn --check
python task_2/run_task_2.py --step leace_cnn -- --device cuda
python task_2/run_task_2.py --step visualize_cnn
```

Il faut les dix checkpoints CNN tous sujets, leurs prédictions, le JSON
d'entraînement et les six archives de cibles Task 2. LEACE vérifie la
provenance, les partitions, les cibles et les probabilités de référence.
Dans chaque pli, la covariance et la transformation sont estimées sur les
moyennes d'essais d'entraînement. La transformation
`f(x) = x - ((x - moyenne_train) @ gauche) @ droite`
est appliquée à chaque token suivant la dimension latente. La moyenne sur
les tokens commute avec cette transformation affine lorsque les poids
sont les mêmes et somment à un. Le CNN reçoit les tokens transformés avec leurs
axes d'origine, puis applique sa normalisation gelée. Cette intervention
vise l'information linéaire associée aux moyennes. Des informations peuvent
subsister dans les contrastes entre canaux ou fenêtres, dans les tokens et
dans les interactions non linéaires.

Les contrôles utilisent le même rang que l'intervention ciblée, le même tirage
gaussien suivi du QR et la même graine `seed + index_du_pli` que le MLP.
Les diagnostics comprennent le rang, la covariance résiduelle d'entraînement,
les sondes Ridge sur les moyennes avant et après projection, le RMS de la
perturbation sur les moyennes et les tokens, la forme et les valeurs finies.
Les scores du sujet tenu à l'écart servent aux diagnostics. Les sondes ne
mesurent pas directement la décodabilité par token.

`results_leace_cnn` contient le JSON, les CSV `by_subject.csv` et `summary.csv`,
puis un dossier par sujet avec les indices, les probabilités de référence,
les facteurs de blanchiment et, par caractéristique, les facteurs ciblés et
probabilités après projection. Les contrôles sont reproductibles à partir
du rang, de la graine, de l'ordre des caractéristiques et du blanchiment.
`--save-embeddings` après `--` sauvegarde uniquement les tokens ciblés des
lignes de test dans des NPY indexés par le `fold.npz` correspondant. Cette
option peut demander plus de 100 Go pour le jeu historique. Les NPY
`*_tokens.npy` sont ignorés par Git sous Task 2.

Le cache décompressé utilise le répertoire temporaire système. Sur le serveur,
`TMPDIR` peut désigner un volume de travail disposant d'assez d'espace.
Les traitements sont réalisés par lots, dont la taille se règle avec
`--batch-size` après `--`. La taille du cache et une borne prudente du
volume des sorties sont vérifiées avant l'extraction.

### Comparaison de résultats existants

```bash
python task_2/run_task_2.py --step compare_models --check
python task_2/run_task_2.py --step compare_models
python task_2/run_task_2.py --step compare_models --classification-only --out task_2/classification/comparaison_classification
```

La comparaison complète exige les résultats LOSO et LEACE des deux modèles.
Elle refuse les cohortes, classes, lignes ou références incompatibles et
signale les différences de rang. Le JSON LEACE MLP historique ne contient
pas les empreintes des six archives de cibles. La comparaison vérifie le
catalogue, la provenance des embeddings et les références sauvegardées.
Les CSV, matrices de confusion, figures et tableaux HTML sont produits à part.
Les chroniques historiques `Delta_*` gardent le signe intervention moins
baseline. Dans les rapports comparatifs, une perte positive indique
une dégradation, en points de pourcentage pour les scores bornés et en
unités de log-loss pour cette dernière. L'effet relatif soustrait la perte
moyenne des contrôles. Pour les familles, les caractéristiques sont d'abord
moyennées dans chaque sujet. Les écarts-types décrivent la variabilité entre
sujets et aucune observation supplémentaire n'est déduite des essais ou des
caractéristiques. La comparaison porte sur deux pipelines qui diffèrent par
le classifieur et l'agrégation des embeddings.
