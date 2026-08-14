"""
ml_pipeline.py
---------------
LEVEL 1 (ML) deliverables for Problem 061 - Personalized Health &
Wellness Coaching for Corporate Employees.

1. Employee segmentation via KMeans clustering (unsupervised).
2. Stress-level prediction via regression (Random Forest Regressor).
3. Burnout-risk classification baseline (Random Forest Classifier) --
   this also becomes the ML baseline that the Level 2 DL model must
   beat on Day/Level comparison.

Run:  python -m src.ml_pipeline   (from the project root)
"""

from pathlib import Path
import json

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from preprocessing import get_train_test_split, build_preprocessor
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_segmentation(preprocessor, X_all, k=4):
    """Employee segmentation: cluster employees on their wellness profile."""
    X_all_transformed = preprocessor.transform(X_all)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_all_transformed)
    sil_score = silhouette_score(X_all_transformed, labels)

    # 2D visualization via PCA
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_all_transformed)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", alpha=0.6, s=15)
    plt.title(f"Employee Wellness Segmentation (KMeans, k={k})\nSilhouette Score: {sil_score:.3f}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "segmentation_clusters.png", dpi=150)
    plt.close()

    joblib.dump(kmeans, MODELS_DIR / "kmeans_segmentation.joblib")
    return labels, sil_score


def run_stress_regression(preprocessor, X_train, X_test, y_train, y_test):
    """Regression: predict continuous stress_level (0-100)."""
    Xt_train = preprocessor.fit_transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(Xt_train, y_train)
    preds = model.predict(Xt_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }

    joblib.dump(model, MODELS_DIR / "stress_regressor.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    return metrics, model


def run_burnout_classification(preprocessor, X_train, X_test, y_train, y_test):
    """Classification baseline: predict binary burnout_risk. Also the
    ML baseline that the Level 2 DL model is compared against."""
    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    model.fit(Xt_train, y_train)

    preds = model.predict(Xt_test)
    probs = model.predict_proba(Xt_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Burnout Classification - Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["No Burnout", "Burnout"])
    plt.yticks([0, 1], ["No Burnout", "Burnout"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "burnout_confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Burnout Classification - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "burnout_roc_curve.png", dpi=150)
    plt.close()

    # Feature importance
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:12]
    plt.figure(figsize=(8, 6))
    plt.barh([feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1])
    plt.title("Top Feature Importances - Burnout Classification")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(model, MODELS_DIR / "burnout_classifier_ml_baseline.joblib")
    return metrics, model


def main():
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = get_train_test_split()
    preprocessor = build_preprocessor()

    print("Fitting preprocessor + running stress regression...")
    reg_metrics, _ = run_stress_regression(preprocessor, X_train, X_test, y_reg_train, y_reg_test)

    print("Running burnout classification baseline...")
    clf_metrics, _ = run_burnout_classification(preprocessor, X_train, X_test, y_clf_train, y_clf_test)

    print("Running employee segmentation (KMeans)...")
    import pandas as pd
    X_all = pd.concat([X_train, X_test], axis=0)
    _, sil_score = run_segmentation(preprocessor, X_all, k=4)

    report = {
        "stress_regression": reg_metrics,
        "burnout_classification_ml_baseline": clf_metrics,
        "segmentation_silhouette_score": sil_score,
    }
    with open(REPORTS_DIR / "ml_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== LEVEL 1 (ML) RESULTS ===")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts saved to: {MODELS_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    main()
