# Autism Screening Prediction (AQ-10)

Machine-learning models that flag who should be referred for a full autism assessment, based on answers to the **AQ-10 Adult** screening questionnaire plus a few background details.

> **Screening aid, not a diagnosis.** A positive result means "consider a full assessment by a qualified clinician". A negative result does not rule autism out.

## What it does

- Cleans the Kaggle *Autism Prediction* dataset (`?` → missing values, inconsistent labels merged).
- Compares three approaches with **5-fold stratified cross-validation**:
  - **AQ-10 rule (total ≥ 6)**: the questionnaire's own cut-off, used as the baseline the ML models must beat.
  - **Logistic Regression**
  - **Random Forest**
- Picks the model with the best PR-AUC and evaluates it once on a held-out 20% test set.
- Reports ASD-class **recall** (share of real cases caught), precision, F1, ROC-AUC and PR-AUC, and saves plots.
- `predict.py` scores a new person's answers from the command line.

All preprocessing (imputation, scaling, one-hot encoding) sits inside scikit-learn `Pipeline`s, so it is fitted on training folds only and nothing leaks from the test data.

## Dataset

Kaggle **Autism Prediction** (`train.csv`, 800 rows). Download it from Kaggle and put it at `data/train.csv`. The CSV is git-ignored, so it isn't committed.

| Column | Meaning |
|---|---|
| `A1_Score` … `A10_Score` | The 10 AQ-10 items, each scored 0/1 (1 = answer leans towards autistic traits) |
| `age`, `gender` | Age in years, `m` / `f` |
| `jaundice` | Born with jaundice (`yes` / `no`) |
| `austim` | Immediate family member diagnosed with autism (`yes` / `no`) |
| `used_app_before` | Has used a screening app before |
| `relation` | Who filled in the form (Self, Parent, …) |
| `ethnicity`, `contry_of_res` | Only used with `--with-demographics` |
| `Class/ASD` | Target (1 = ASD) |

`ID`, `result` (≈ the AQ-10 total, redundant with A1–A10) and `age_desc` (a constant) are not used as features.

The Kaggle data is synthetic, generated from the UCI AQ-10 screening data. Treat the numbers as a learning exercise, not clinical evidence.

### The AQ-10 items

| Item | Asks about | Scores 1 if the person… |
|---|---|---|
| A1 | Noticing small sounds others don't | agrees |
| A2 | Focusing on the whole picture rather than details | disagrees |
| A3 | Doing more than one thing at once | disagrees |
| A4 | Getting back to a task after an interruption | disagrees |
| A5 | Reading between the lines | disagrees |
| A6 | Telling when a listener is bored | disagrees |
| A7 | Working out characters' intentions in stories (finds it hard) | agrees |
| A8 | Collecting information about categories of things | agrees |
| A9 | Reading thoughts and feelings from faces | disagrees |
| A10 | Working out people's intentions (finds it hard) | agrees |

Each item is answered on a 4-point scale (definitely/slightly agree/disagree). "Agrees" means either agree option, and "disagrees" means either disagree option. A total of **6 or more** is the referral threshold. Get the official wording from the Autism Research Centre, Cambridge (autismresearchcentre.com → Tests). The AQ-10 Adult is meant for ages 16+, and separate versions exist for adolescents and children.

## How to run

### 1. Set up

```bash
git clone https://github.com/aastha-m22/Autism-Prediction-System.git
cd Autism-Prediction-System
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download `train.csv` from Kaggle into the `data/` folder.

### 2. Train and evaluate

```bash
python train.py
```

This prints the cross-validation table and the test-set report, and writes:

- `outputs/cv_results.csv`: CV metrics for every model
- `outputs/test_metrics.json`: final test metrics
- `outputs/confusion_matrix.png`, `outputs/pr_roc_curves.png`, `outputs/feature_importance.png`
- `models/autism_model.joblib`: the trained model used by `predict.py`

To check whether ethnicity and country of residence change anything:

```bash
python train.py --with-demographics --out outputs_demographics
```

### 3. Screen a new person

Interactive (asks for each answer):

```bash
python predict.py
```

Or with flags:

```bash
python predict.py --scores 1,0,1,1,0,1,1,0,1,1 --age 24 --gender f \
    --jaundice no --family-history no --used-app-before no --relation Self
```

### 4. Notebook

```bash
jupyter notebook Autism_Prediction.ipynb
```

The notebook covers the same pipeline with EDA and inline plots. It uses `autism_model.py`, so it always matches the scripts. In Google Colab, uncomment the setup cell at the top.

## Results

Run `python train.py` and fill this in from `outputs/cv_results.csv` and `outputs/test_metrics.json`.

| Model | CV recall (ASD) | CV precision | CV PR-AUC |
|---|---|---|---|
| AQ-10 rule (total ≥ 6) | | | |
| Logistic Regression | | | |
| Random Forest | | | |

## What changed from the first version

- **Fixed swapped metrics.** `classification_report(pred, y_test)` had its arguments reversed, so ASD recall and precision were printed the wrong way round.
- **Removed `ID` as a feature.** The row number was being fed into the model.
- **Age is now actually used.** Previously `age` was dropped and `Age_Cat` was overwritten with NaNs.
- **No target leakage.** Categories used to be encoded by their target mean across the whole dataset, before splitting. They're now one-hot encoded inside the pipeline.
- **Proper evaluation.** Stratified 5-fold CV, fixed random seeds, a held-out test set, PR-AUC, and a questionnaire-rule baseline.
- **Class imbalance** is handled with class weights instead of SMOTEENN, which is simpler and doesn't need resampling inside CV.
- **Runs outside Colab** with a `requirements.txt`, reusable modules and a CLI predictor.

## Project structure

```
autism_model.py          data cleaning, features, models, CV helper (shared)
train.py                 train, compare, evaluate, save model + plots
predict.py               score one person's answers
Autism_Prediction.ipynb  walkthrough notebook with EDA
data/                    put Kaggle train.csv here (git-ignored)
```

## Future work

- Choose the decision threshold for a target recall (for example, catch at least 90% of cases).
- Calibrate probabilities.
- Validate on real (non-synthetic) screening data.
- Add a simple web front-end (Streamlit).
