"""
dl_pipeline.py
---------------
LEVEL 2 (DL) deliverable for Problem 061 - Personalized Health &
Wellness Coaching for Corporate Employees.

Uses the 12-week behavioral tracking data (sleep, activity, work
hours, mood, weekly stress) per employee as a sequence, and trains an
LSTM + Attention classifier to predict burnout_risk -- capturing
temporal/behavioral change that the Level 1 cross-sectional ML model
cannot see.

Requires TensorFlow (see requirements.txt). Run:
    python -m src.dl_pipeline

NOTE: this script was authored and unit-checked for correctness in an
offline environment without TensorFlow installed, so run it locally
first with `pip install -r requirements.txt` before training.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "employee_weekly_tracking.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_FEATURES = [
    "sleep_hours", "physical_activity_min", "work_hours", "mood_score", "weekly_stress_level"
]
N_WEEKS = 12
RANDOM_SEED = 42


def load_sequences():
    df = pd.read_csv(RAW_PATH)
    employees = df["employee_id"].unique()

    X, y = [], []
    for emp_id in employees:
        emp_df = df[df["employee_id"] == emp_id].sort_values("week")
        if len(emp_df) != N_WEEKS:
            continue
        X.append(emp_df[SEQUENCE_FEATURES].values)
        y.append(emp_df["burnout_risk"].iloc[0])

    X = np.array(X, dtype=np.float32)   # shape: (n_employees, N_WEEKS, n_features)
    y = np.array(y, dtype=np.int32)
    return X, y


def scale_sequences(X_train, X_test):
    """Standardize each feature channel using train-set statistics only
    (fit on train, applied to test -- avoids data leakage)."""
    n_train, t, f = X_train.shape
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, f)
    scaler.fit(X_train_2d)

    X_train_scaled = scaler.transform(X_train_2d).reshape(n_train, t, f)
    X_test_scaled = scaler.transform(X_test.reshape(-1, f)).reshape(X_test.shape[0], t, f)
    return X_train_scaled, X_test_scaled, scaler


def build_lstm_attention_model(n_timesteps: int, n_features: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(n_timesteps, n_features), name="behavioral_sequence")

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.2))(inputs)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True, dropout=0.2))(x)

    # Simple additive attention over the timestep axis
    attention_scores = layers.Dense(1, activation="tanh")(x)
    attention_weights = layers.Softmax(axis=1, name="attention_weights")(attention_scores)
    context_vector = layers.Multiply()([x, attention_weights])
    context_vector = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(context_vector)

    x = layers.Dense(32, activation="relu")(context_vector)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="burnout_risk")(x)

    model = models.Model(inputs, outputs, name="lstm_attention_burnout_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def flatten_baseline_metrics(X_train, X_test, y_train, y_test):
    """Quick same-data ML baseline (flattened sequence -> Random Forest)
    so the DL model's lift over ML is measured on identical data/split,
    not just against the Level 1 cross-sectional report."""
    from sklearn.ensemble import RandomForestClassifier

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_SEED, class_weight="balanced")
    clf.fit(X_train_flat, y_train)
    preds = clf.predict(X_test_flat)
    probs = clf.predict_proba(X_test_flat)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }


def main():
    print("Loading behavioral sequences...")
    X, y = load_sequences()
    print(f"Loaded {X.shape[0]} employee sequences of shape {X.shape[1:]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train_scaled, X_test_scaled, scaler = scale_sequences(X_train, X_test)

    print("Training flattened-ML baseline on identical split...")
    ml_baseline_metrics = flatten_baseline_metrics(X_train_scaled, X_test_scaled, y_train, y_test)

    print("Building and training LSTM + Attention model...")
    model = build_lstm_attention_model(n_timesteps=N_WEEKS, n_features=len(SEQUENCE_FEATURES))
    model.summary()

    early_stop = callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    probs = model.predict(X_test_scaled).ravel()
    preds = (probs >= 0.5).astype(int)

    dl_metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
    }

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train_loss")
    axes[0].plot(history.history["val_loss"], label="val_loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history.history["auc"], label="train_auc")
    axes[1].plot(history.history["val_auc"], label="val_auc")
    axes[1].set_title("ROC-AUC"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "dl_training_curves.png", dpi=150)
    plt.close()

    # Comparison chart: ML baseline vs DL model
    metrics_to_plot = ["accuracy", "f1_score", "roc_auc"]
    ml_vals = [ml_baseline_metrics[m] for m in metrics_to_plot]
    dl_vals = [dl_metrics[m] for m in metrics_to_plot]
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, ml_vals, width, label="ML Baseline (Random Forest)")
    plt.bar(x + width / 2, dl_vals, width, label="DL Model (LSTM + Attention)")
    plt.xticks(x, metrics_to_plot)
    plt.ylim(0, 1)
    plt.title("Level 2 (DL) vs Level 1 (ML) - Burnout Risk Prediction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "ml_vs_dl_comparison.png", dpi=150)
    plt.close()

    model.save(MODELS_DIR / "lstm_attention_burnout_model.keras")

    report = {
        "ml_baseline_flattened_sequence": ml_baseline_metrics,
        "dl_lstm_attention": dl_metrics,
        "sequence_length_weeks": N_WEEKS,
        "features_used": SEQUENCE_FEATURES,
    }
    with open(REPORTS_DIR / "dl_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== LEVEL 2 (DL) RESULTS ===")
    print(json.dumps(report, indent=2))
    print(f"\nArtifacts saved to: {MODELS_DIR} and {REPORTS_DIR}")


if __name__ == "__main__":
    main()
