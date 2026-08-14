# Personalized Health & Wellness Coaching for Corporate Employees
### Problem 061 — Level 1 (ML) + Level 2 (DL)

This project is a complete, runnable implementation of the **ML** and **DL** levels
from the problem brief: employee wellness segmentation, stress-level prediction,
burnout-risk classification, and a temporal LSTM+Attention model that captures
behavioral change over time.

---

## 1. Setup (VS Code)

```bash
# 1. Open this folder in VS Code:  File > Open Folder...

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (VS Code) select the venv as your Python interpreter:
#    Ctrl/Cmd+Shift+P -> "Python: Select Interpreter" -> ./venv
```

## 2. Run everything

```bash
python main.py --stage all
```

Or run stages individually:

```bash
python main.py --stage data   # generates synthetic datasets in data/raw/
python main.py --stage ml     # Level 1: segmentation + regression + classification
python main.py --stage dl     # Level 2: LSTM + Attention burnout model (needs TensorFlow)
```

Launch the demo app once the ML stage has produced saved models:

```bash
streamlit run app/streamlit_app.py
```

Open the EDA notebook in VS Code's Jupyter extension:

```
notebooks/01_EDA.ipynb
```

---

## 3. Folder structure

```
health_wellness_project/
├── README.md
├── requirements.txt
├── main.py                      <- single entry point (see stages above)
├── data/
│   ├── raw/                     <- generated CSVs (swap in real data here)
│   └── processed/
├── src/
│   ├── data_generation.py       <- synthetic dataset generator
│   ├── preprocessing.py         <- cleaning, feature engineering, train/test split
│   ├── ml_pipeline.py           <- LEVEL 1: clustering + regression + classification
│   └── dl_pipeline.py           <- LEVEL 2: LSTM + Attention model
├── notebooks/
│   └── 01_EDA.ipynb             <- exploratory data analysis
├── app/
│   └── streamlit_app.py         <- interactive demo UI
├── models/                      <- saved .joblib / .keras models (created on run)
└── reports/                     <- metrics (.json) + plots (.png), created on run
```

Swap `data/raw/employee_wellness_cross_sectional.csv` and
`employee_weekly_tracking.csv` for real HR/wellness exports whenever available —
every downstream script depends only on the column names, not on how the data
was produced.

---

## 4. What each level delivers (mapped to the brief's evaluation rubric)

### LEVEL 1 — Machine Learning (`src/ml_pipeline.py`)

| Deliverable from brief | Where it lives |
|---|---|
| Baseline ML model (clustering for segmentation, regression for stress) | `run_segmentation()`, `run_stress_regression()` |
| Data preprocessing pipeline | `src/preprocessing.py` (imputation, scaling, one-hot encoding via `ColumnTransformer`) |
| Feature engineering report | `reports/feature_engineering_report.json` + engineered features: `activity_index`, `recovery_index`, `lifestyle_risk_score` |
| Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC) | `reports/ml_evaluation_report.json`, `reports/burnout_confusion_matrix.png`, `reports/burnout_roc_curve.png` |

Sample results from a local run (yours may vary slightly with each synthetic seed):
- Stress regression: RMSE ≈ 9.5, R² ≈ 0.43
- Burnout classification baseline: Accuracy ≈ 0.70, ROC-AUC ≈ 0.74
- Segmentation: 4 employee wellness clusters, visualized via PCA in `reports/segmentation_clusters.png`

### LEVEL 2 — Deep Learning (`src/dl_pipeline.py`)

| Deliverable from brief | Where it lives |
|---|---|
| DL model (LSTM/attention for temporal behavior) | `build_lstm_attention_model()` — Bidirectional LSTM stack + additive attention over 12 weekly timesteps |
| Model architecture design | Same function; architecture rationale in the docstring |
| Hyperparameter tuning report | Extend `dl_pipeline.py`'s `main()` — currently uses documented defaults (see comments); loop over `learning_rate`/`units` for a tuning sweep if your rubric requires a formal search |
| Performance comparison with ML baseline | `reports/ml_vs_dl_comparison.png` + `reports/dl_evaluation_report.json` (same train/test split, flattened-sequence Random Forest vs. LSTM+Attention) |

The DL script needs TensorFlow, which isn't available in the environment this
project was authored in — the code was written and reviewed for correctness,
but **run it locally first** (`pip install -r requirements.txt` already
includes TensorFlow) to confirm training on your machine before submitting.

---

## 5. Common pitfalls this project already guards against

- **Data leakage**: `StandardScaler`/imputers are fit on the *train* split only, then applied to test (see `preprocessing.py` and `dl_pipeline.py::scale_sequences`).
- **Class imbalance**: `class_weight="balanced"` on both classifiers; stratified train/test split.
- **Duplicate rows**: `load_raw()` explicitly deduplicates on `employee_id` and logs how many were removed.
- **Overfitting (DL)**: dropout in every LSTM/Dense layer + early stopping on validation AUC.

---

## 6. Next steps toward Level 3+ (not included here)

The original 9-day framework continues into NLP (employee feedback/journal
analysis), SLM (empathetic message generation), LLD (system architecture),
GenAI (personalized content generation), and Agentic AI (autonomous wellness
coach). This project intentionally stops at ML + DL per your current request —
ask if you'd like the same treatment for any of the later levels.
