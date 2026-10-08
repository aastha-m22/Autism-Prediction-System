"""Train and evaluate autism screening models.

Usage:
    python train.py                       # uses data/train.csv
    python train.py --data path/to/train.csv
    python train.py --with-demographics   # also use ethnicity + country

Outputs:
    outputs/cv_results.csv     cross-validated metrics for every model
    outputs/test_metrics.json  held-out test metrics for the selected model
    outputs/*.png              confusion matrix, PR/ROC curves, feature importance
    models/autism_model.joblib the trained model + metadata used by predict.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from autism_model import SCORING, SEED, build_models, load_data, run_cv, split_xy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/train.csv", help="path to the Kaggle train.csv")
    parser.add_argument("--with-demographics", action="store_true",
                        help="include ethnicity and country of residence as features")
    parser.add_argument("--out", default="outputs", help="folder for metrics and plots")
    parser.add_argument("--model-path", default="models/autism_model.joblib")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    X, y = split_xy(df, args.with_demographics)
    print(f"Loaded {len(df)} rows; class balance: {y.value_counts().to_dict()}")

    # Hold out a test set first. Everything below (CV, model choice) only
    # ever sees the training part.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    models = build_models(args.with_demographics, random_state=SEED)

    print("\n5-fold cross-validation on the training set:")
    cv_results = run_cv(models, X_train, y_train)
    cv_results.to_csv(out / "cv_results.csv")
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(cv_results[list(SCORING)])

    # Pick the learned model with the best PR-AUC. The rule is the baseline
    # it has to beat, not a candidate.
    learned = cv_results.drop(index="AQ-10 rule (total >= 6)")
    best_name = learned["pr_auc"].idxmax()
    baseline = cv_results.loc["AQ-10 rule (total >= 6)"]
    print(f"\nSelected model: {best_name}")
    gain = learned.loc[best_name, "pr_auc"] - baseline["pr_auc"]
    print(f"PR-AUC vs AQ-10 rule baseline: {gain:+.3f}")

    model = models[best_name].fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    # y_true first, y_pred second (the original notebook had these swapped).
    print("\nHeld-out test set:")
    print(classification_report(y_test, pred, target_names=["No ASD", "ASD"], digits=3))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix [rows = true, cols = predicted]:\n", cm)

    report = classification_report(y_test, pred, output_dict=True)
    test_metrics = {
        "model": best_name,
        "with_demographics": args.with_demographics,
        "accuracy": report["accuracy"],
        "asd_recall": report["1"]["recall"],
        "asd_precision": report["1"]["precision"],
        "asd_f1": report["1"]["f1-score"],
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "confusion_matrix": cm.tolist(),
        "baseline_rule_test": {
            "asd_recall": classification_report(
                y_test, models["AQ-10 rule (total >= 6)"].fit(X_train).predict(X_test), output_dict=True
            )["1"]["recall"],
        },
    }
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

    # --- Plots -----------------------------------------------------------
    ConfusionMatrixDisplay(cm, display_labels=["No ASD", "ASD"]).plot(cmap="Blues")
    plt.title(f"{best_name} - test set")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1], name=best_name)
    axes[1].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("Precision-Recall (test)")
    axes[1].set_title("ROC (test)")
    plt.tight_layout()
    plt.savefig(out / "pr_roc_curves.png", dpi=150)
    plt.close()

    # Permutation importance on the raw input columns (model-agnostic, and
    # computed on held-out data so it is not inflated by overfitting).
    imp = permutation_importance(
        model, X_test, y_test, scoring="average_precision", n_repeats=20, random_state=SEED
    )
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
    importance.to_csv(out / "feature_importance.csv", header=["importance"])
    plt.figure(figsize=(8, 6))
    importance.plot.barh(color=sns.color_palette("crest", len(importance)))
    plt.xlabel("Drop in PR-AUC when the feature is shuffled")
    plt.title("Permutation feature importance (test)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(
        {
            "model": model,
            "model_name": best_name,
            "features": list(X.columns),
            "with_demographics": args.with_demographics,
        },
        args.model_path,
    )
    print(f"\nSaved model to {args.model_path} and results to {out}/")


if __name__ == "__main__":
    main()
