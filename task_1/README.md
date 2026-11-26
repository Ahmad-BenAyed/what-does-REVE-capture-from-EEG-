# Task 1 — Caractéristiques EEG dans les embeddings REVE

L’analyse principale porte sur la version **C**, qui mesure dans quelle mesure les
embeddings suivent les variations EEG **intra-sujet**. Les versions **A** et **B**
servent de comparaisons. [Ouvrir le tableau de bord autonome](visualisations/dashboard.html).

| Version | Cibles EEG | Embeddings REVE |
|---|---|---|
| A | Brutes | Bruts |
| B | Z-score par sujet | Bruts |
| C | Z-score par sujet | Z-score par sujet |

En C, le z-score des cibles et des embeddings utilise **l’ensemble des essais du
sujet testé**. Cette normalisation est **transductive** et la conclusion porte
sur les variations intra-sujet. Une généralisation inductive vers un nouveau
sujet sans calibration demanderait un autre protocole. B utilise également les
statistiques des cibles du sujet testé.

## Protocole

Le script [loso_normalisation_sujet.py](loso_normalisation_sujet.py) réunit les six
familles T (temporelle), F (spectrale), TF (temps-fréquence), C (complexité),
R (connectivité) et X (couplage interfréquence). L’exécution de référence contient
2 642 essais, 10 sujets et 63 caractéristiques. Le mean pooling sur les canaux et
les fenêtres donne 512 dimensions d’embedding. Les 308 cibles correspondent à
49 caractéristiques résumées sur les canaux par Mean, Standard deviation,
Median, Q25, Q75 et Max, et à 14 caractéristiques globales R. Aucune ligne
non finie n’a été retirée dans cette exécution.

Le LOSO externe tient un sujet à l’écart. Dans chaque pli, le LOSO interne
sélectionne un alpha Ridge par cible selon le R² moyen sur les sujets internes.
La standardisation des prédicteurs utilisée par Ridge est ajustée sur
l’entraînement du pli. Les scores externes comprennent Pearson et R², avec la
moyenne de la cible du sujet testé comme référence du R². Les résumés agrègent
d’abord les dimensions par caractéristique au sein de chaque sujet, puis les
caractéristiques par famille au sein du sujet, puis les sujets. Les scores
indéfinis sont exclus des moyennes concernées. Les R² négatifs sont conservés.
Les axes R² combinant des valeurs extrêmes utilisent une échelle symlog,
linéaire autour de zéro.

Les cinq permutations par version mélangent les lignes des cibles conjointement
au sein de chaque sujet. Les figures montrent les valeurs individuelles, leur
moyenne et leur maximum comme contrôles descriptifs. Cinq permutations donnent
une résolution insuffisante pour établir une significativité forte.

Le décodage d’identité par régression logistique à cinq plis stratifiés donne
100 % de balanced accuracy sur les embeddings bruts, environ 3,9 % après
normalisation par sujet et 10 % au hasard. Le contrôle mesure la décodabilité
avec ce modèle et ce protocole. Une réduction du score ne démontre pas une
suppression complète de l’information d’identité.

## Résultats et figures

Les fichiers de l’exécution de référence se trouvent dans
[results_loso_norm_sujet](results_loso_norm_sujet). Le visualiseur contrôle la
grille des scores, les alphas et la cohérence des deux résumés. Il lit les
métriques dans les CSV d’origine. Le cache local `pooled_embeddings.npy` est
ignoré par Git. Sa forme est contrôlée s’il est présent et les CSV suffisent
pour régénérer les figures.
Le fichier publié `config.portable.json` conserve les paramètres de référence
et exprime les chemins d’entrée relativement au dossier des résultats.
Le `config.json` original est conservé localement et ignoré par Git. Le PNG
`comparaison_familles.png` est un export original conservé avec les résultats.

| Figures | Contenu |
|---|---|
| 01–02 | R² et Pearson des 63 caractéristiques et boxplots par famille, version C |
| 03 | Pearson contre R² sur les plis externes dans l’exécution de référence |
| 04 | Les 14 caractéristiques R, comparaison A/B/C |
| 05 | Caractéristiques × statistiques et caractéristiques × sujets en C, R² et Pearson |
| 06 | Comparaisons A/B/C par caractéristique et famille, R² moyen et médian, Pearson et dispersion entre sujets |
| 07–08 | Permutations par version, famille et caractéristique, puis décodage de l’identité |
| 09 | R² et Pearson par sujet et famille pour A, B et C |

Les six statistiques de la figure 05 résument les canaux pour chaque essai.
Pour R, la colonne représente une cible globale. Les scores internes sont
absents des CSV de référence. La figure 03 présente donc les deux métriques
externes. Les nouvelles exécutions écrivent `validation_interne.csv` pour les
cibles réelles, avec le R² interne moyen à l’alpha retenu pour chaque cible et
pli externe. Lorsque ce fichier est présent, la figure 03 compare automatiquement
ce R² interne au R² de test externe.

Les anciens scripts par famille, résultats et figures sont récupérables dans
Git au commit `21a7bfc` sous `task_1/family` et `task_1/visualisations`.
Par exemple, `git show 21a7bfc:task_1/visualiser_resultats.py` affiche l’ancien
visualiseur.

## Commandes PowerShell

Depuis la racine du dépôt, vérifier les résultats et régénérer les exports PNG,
PDF et HTML à partir des fichiers existants.

```powershell
python -m pip install -r task_1/requirements.txt
python -m task_1.run_task_1 visualiser --check
python -m task_1.run_task_1 visualiser
Invoke-Item task_1/visualisations/dashboard.html
python -m unittest discover -s task_1/tests
```

Les chemins par défaut sont définis dans `task_1/config.json`, relativement au
dossier de ce fichier. Les entrées par défaut
se trouvent dans `data/` à la racine du dépôt. Pour une nouvelle exécution, fournir les métadonnées,
les embeddings et `features_63.npz` (produit par
[`legacy/03_extract_63_features.py`](../legacy/03_extract_63_features.py)).
Le contrôle d’entrée vérifie la présence des fichiers et la disponibilité de la
sortie. Le pipeline refuse un dossier d’expérience déjà rempli.

```powershell
python -m task_1.check_inputs --out task_1/results_loso_nouveau
python -m task_1.run_task_1 experiment --out task_1/results_loso_nouveau
python -m task_1.run_task_1 visualiser --results task_1/results_loso_nouveau --out task_1/visualisations_nouvelles
```

Les options `--metadata`, `--features`, `--embeddings` et `--config` permettent
d’indiquer d’autres chemins. L’option `--pooled` réutilise un cache du même jeu
d’essais dans le même ordre. Le cache d’origine est en float32, tandis que le
pooling à partir du NPZ utilise float64 dans le calcul.

L’expérience facultative ci-dessous ajuste une cible par indice de canal pour
chacune des 49 caractéristiques concernées et conserve les 14 cibles globales
R. Elle exige des dossiers de sortie distincts et peut être nettement plus
coûteuse. Les figures portent alors explicitement sur les indices de canaux EEG.

```powershell
python -m task_1.run_task_1 experiment --par-canal --out task_1/results_loso_canaux
python -m task_1.run_task_1 visualiser --results task_1/results_loso_canaux --out task_1/visualisations_canaux
```
