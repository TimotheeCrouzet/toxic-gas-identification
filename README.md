# Challenge ML

Projet de travail pour le challenge `Toxic gas identification`.

## Architecture Du Projet

### Donnees

- `data/raw/`
  Donnees source du challenge :
  - `x_train.csv`
  - `y_train.csv`
  - `x_test.csv`

- `data/processed/`
  Toutes les sorties du pipeline :
  - soumissions CSV
  - fichiers intermediaires eventuels

  Note :
- les fichiers de donnees brutes ne sont pas inclus dans l'archive rendue afin d'en limiter la taille ;
- les scripts supposent toutefois la presence de `data/raw/x_train.csv`, `data/raw/y_train.csv` et `data/raw/x_test.csv` ;
- sans ces fichiers, le pipeline ne peut pas etre execute.


### Code

- [src/challenge_ml/feature.py](src/challenge_ml/feature.py)
  Preparation des donnees.

  Fonctions importantes :
  - `load_datasets()`
  - `feature_engineering(features)`
  - `build_training_data(use_features=True)`

  Role :
  - charger les CSV ;
  - preparer les variables brutes ou enrichies ;
  - construire `X_train`, `y_train`, `X_test`, `test_ids`.

- [src/challenge_ml/split.py](src/challenge_ml/split.py)
  Logique de validation locale.

  Fonctions importantes :
  - `random_split(...)`
  - `humidity_split(...)`
  - `random_split_from_features(...)`
  - `humidity_split_from_features(...)`

  Role :
  - fabriquer un split aleatoire ;
  - fabriquer un split oriente `Humidity`.

- [src/challenge_ml/train_xgboost.py](src/challenge_ml/train_xgboost.py)
  Pipeline principal d'entrainement, d'evaluation et de soumission.

  Fonctions importantes :
  - `build_xgb_model(...)`
  - `evaluate_model(...)`
  - `compare_raw_vs_features()`
  - `run_regularized_feature_model()`
  - `train_full_and_predict(...)`
  - `build_raw_submission()`
  - `build_regularized_feature_submission()`

## Fonctionnement Du Pipeline

Le projet a deux usages distincts.

### 1. Evaluation Locale

Utiliser les splits pour comparer plusieurs idees avant soumission.

Splits disponibles :
- `random`
- `humidity`

Ces splits servent a :
- comparer `raw` vs `features`
- tester une version plus regularisee
- mesurer localement la robustesse

### 2. Soumission Finale

Une fois un modele choisi :
- on entraine sur tout `x_train` / `y_train`
- on predit sur `x_test`
- on genere un CSV dans `data/processed`

C'est le role de :
- `train_full_and_predict(...)`

## Commandes A Executer

### Installer L'Environnement

```bash
uv sync
uv pip install -e .
```

Ces deux commandes :
- installent les dependances ;
- installent le package local `challenge_ml` en mode editable ;
- permettent aux imports du type `from challenge_ml...` de fonctionner correctement.

### Lancer Les Experiences Locales

Depuis la racine du projet :

```bash
uv run python -m challenge_ml.train_xgboost
```

Cette commande lance :
- la comparaison `raw` vs `features`
- le modele `features regularized`
- un ensemble local simple

### Generer Une Soumission `raw`

```bash
uv run python -c "from challenge_ml.train_xgboost import build_raw_submission; build_raw_submission()"
```

Fichier généré :
- `data/processed/submission_xgboost_raw.csv`

### Generer Une Soumission `features regularized`

```bash
uv run python -c "from challenge_ml.train_xgboost import build_regularized_feature_submission; build_regularized_feature_submission()"
```

Fichier genere :
- `data/processed/submission_xgboost_features_regularized.csv`

### Variante Sans `uv`

Si besoin, une installation plus classique est aussi possible :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Imports Utiles

### Charger Les Donnees Preparees

```python
from challenge_ml.feature import build_training_data

X_train, y_train, X_test, test_ids = build_training_data(use_features=True)
```

### Utiliser Un Split Local

```python
from challenge_ml.split import random_split_from_features, humidity_split_from_features

X_train, X_valid, y_train, y_valid = random_split_from_features(use_features=True)
```

ou

```python
X_train, X_valid, y_train, y_valid = humidity_split_from_features(use_features=False)
```

### Generer Une Soumission Depuis Python

```python
from challenge_ml.train_xgboost import build_raw_submission

model, submission = build_raw_submission()
```

## Format D'Une Soumission

Le CSV final doit contenir :
- `ID`
- `c01` a `c23`
- `c15` remise a `0.0`

Le pipeline gere cela automatiquement.
