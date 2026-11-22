# MLP compact sur tous les sujets

Ce patch ajoute un MLP comme nouvelle expérience de classification sur les embeddings
REVE. Il inclut **tous les sujets par défaut**, y compris sub-P02, et conserve les
résultats du CNN. Le MLP ne garantit pas une meilleure généralisation ; les faibles
scores du CNN ne démontrent pas à eux seuls que sa taille était la cause.

## Architecture

Pour une entrée `N × 41 × 8 × 512` :

1. Moyenne fixe des 41 canaux et des 8 fenêtres, séparément pour chaque segment :
   vecteur de 512 valeurs.
2. Standardisation de ces vecteurs : moyenne et écart-type calculés uniquement sur
   les lignes d'entraînement du pli concerné.
3. Couche dense 512 → 64, ReLU, dropout 25 %.
4. Couche dense 64 → 3 ; cross-entropy pour l'entraînement, softmax à la prédiction.

Cela représente **33 027 paramètres** pour trois classes, contre 101 827 pour le
CNN fourni. Le pooling est calculé une seule fois avant les entraînements, ce qui
réduit leur coût et la représentation en RAM. Les indices de lignes sont conservés.

Cette comparaison change à la fois le classifieur et l'agrégation : le CNN conserve
les électrodes et traite les fenêtres, tandis que ce MLP utilise leur moyenne.
Elle compare deux pipelines et n'isole pas l'effet de la seule architecture.

## Installation dans le dépôt déjà équipé du CNN

Télécharger `mlp-reve-loso.patch`, puis depuis le terminal de VS Code à la racine :

```powershell
git apply --check "C:\Users\Ahmad\Downloads\mlp-reve-loso.patch"
git apply "C:\Users\Ahmad\Downloads\mlp-reve-loso.patch"
python -m unittest discover -s tests -p test_mlp_reve.py -v
```

Les dépendances sont celles du CNN. Si nécessaire :

```powershell
python -m pip install -r classification\requirements.txt
```

## Entraîner sur les dix sujets

Avec les trois fichiers d'entrée dans `..\data` :

```powershell
python -u -m classification.train_mlp_loso
```

**Aucune option d'exclusion n'est nécessaire.** Les valeurs par défaut sont :

- tous les sujets ;
- LOSO externe, donc dix plis pour les données actuelles ;
- trois plis de validation interne GroupKFold par sujet ;
- au plus vingt époques, patience cinq ;
- choix de la durée par la médiane des meilleures époques de validation ;
- nouvel entraînement sur tous les sujets d'entraînement externe avant le test ;
- AdamW, learning rate 0,001, weight decay 0,001 ;
- lots de seize ; CPU ou CUDA selon disponibilité.

Le protocole de sélection de durée et les graines sont identiques à ceux du CNN.
Les performances externes ne servent pas à sélectionner l'époque. Cette nouvelle
architecture a été proposée après lecture des résultats du CNN : conserver la
trace de cette comparaison exploratoire.

Les chemins `--embeddings`, `--segments`, `--metadata`, `--out`, ainsi que les autres
options du CNN restent disponibles. `--width` règle la couche cachée du MLP.
Les filtres de lignes non finies sont appliqués avant le pooling. Le NPZ original
et ses labels ne sont pas modifiés.

## Résultats

Le dossier par défaut est :

```text
classification/results_mlp_tous_sujets/
```

Il contient les mêmes fichiers de résultats que le CNN :
`fold_metrics.csv`, `metrics_summary.csv`, `predictions.csv`, `outer_splits.csv`,
`rows_qc.csv`, `class_counts_by_subject.csv`, `config.json`, et un répertoire
`fold_00`, etc., pour chaque sujet testé. Chaque pli sauvegarde son `checkpoint.pt`,
sa normalisation, ses indices train/test, ses historiques et sa matrice de confusion.
Le `config.json` distingue la forme source complète de la forme utilisée après
pooling. Les anciens checkpoints CNN restent lisibles.

Afficher la moyenne et l'écart-type entre sujets :

```powershell
python -c "import pandas as pd; print(pd.read_csv(r'classification\results_mlp_tous_sujets\metrics_summary.csv',index_col=0).to_string())"
```

Afficher les résultats par sujet :

```powershell
python -c "import pandas as pd; d=pd.read_csv(r'classification\results_mlp_tous_sujets\fold_metrics.csv'); print(d[['Subject','Accuracy','Balanced_Accuracy','Macro_F1','Majority_Accuracy','Selected_Epoch']].to_string(index=False))"
```

Comparer les moyennes au CNN déjà entraîné sur tous les sujets :

```powershell
python -c "import pandas as pd; a=pd.read_csv(r'classification\results_cnn_tous_sujets\metrics_summary.csv',index_col=0); b=pd.read_csv(r'classification\results_mlp_tous_sujets\metrics_summary.csv',index_col=0); print(pd.DataFrame({'CNN':a.loc['mean'],'MLP':b.loc['mean']}).to_string())"
```

Une différence de moyenne seule n'établit pas sa significativité. Les sorties par
sujet permettent une comparaison appariée si les cohortes et les lignes sont identiques.

## Reprise et test rapide

Après interruption, avec les mêmes options d'entraînement :

```powershell
python -u -m classification.train_mlp_loso --resume
```

Les plis terminés sont conservés ; un pli incomplet repart du début. Une reprise
CNN/MLP croisée est refusée. Un lancement sans `--resume` refuse un dossier non vide.

Pour une vérification courte sur un seul sujet, dans un dossier distinct :

```powershell
python -u -m classification.train_mlp_loso --epochs 2 --inner-folds 2 --test-subjects sub-P01 --out classification\results_mlp_verification
```

Ce test partiel n'est pas une évaluation LOSO complète.

## Réutiliser le MLP gelé pour l'effacement

La commande suivante recharge le MLP du premier pli et reproduit sa référence,
sans apprentissage :

```powershell
python -m classification.evaluate_frozen_mlp --checkpoint classification\results_mlp_tous_sujets\fold_00\checkpoint.pt --embeddings ..\data\reve_embeddings.npz --metadata ..\data\segments_metadata.csv --out classification\results_mlp_relecture_fold00
```

Plus tard, utiliser le même checkpoint avec les embeddings après effacement :

```powershell
python -m classification.evaluate_frozen_mlp --checkpoint classification\results_mlp_tous_sujets\fold_00\checkpoint.pt --embeddings ..\data\reve_embeddings_effaces.npz --metadata ..\data\segments_metadata.csv --out classification\results_mlp_effacement_fold00
```

Pour le MLP, le nouveau NPZ peut contenir soit les tenseurs complets, soit les
vecteurs moyens `N × 512`, avec la même clé `embeddings`. Il doit conserver toutes
les lignes dans l'ordre original et les labels `y`. Le même CSV metadata est requis.
La commande utilise seulement les indices de test sauvegardés. Les poids et les
statistiques de normalisation restent fixes ; l'effacement intervient avant cette
normalisation. La transformation d'effacement devra être estimée uniquement sur les
indices d'entraînement du checkpoint. Ce patch n'effectue pas encore cet effacement.
