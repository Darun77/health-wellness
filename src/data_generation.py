"""
data_generation.py
-------------------
Generates two synthetic but realistic datasets for the
"Personalized Health & Wellness Coaching for Corporate Employees" project.

1. employee_wellness_cross_sectional.csv
   -> One row per employee. Used for Level 1 (ML): clustering-based
      employee segmentation + regression/classification for stress
      and burnout prediction.

2. employee_weekly_tracking.csv
   -> 12 weeks of behavioral tracking per employee (long format).
      Used for Level 2 (DL): LSTM/Attention model to capture temporal
      behavioral change and predict burnout risk.

Replace this generator with a real HR/wellness data export whenever
one becomes available -- every downstream script only depends on the
column names defined here, not on how the values were produced.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
N_EMPLOYEES = 1200
N_WEEKS = 12

DEPARTMENTS = ["Engineering", "Sales", "Operations", "HR", "Finance", "Support"]


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def generate_cross_sectional(n=N_EMPLOYEES, seed=RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 60, size=n)
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.52, 0.45, 0.03])
    department = rng.choice(DEPARTMENTS, size=n)
    tenure_years = _clip(rng.normal(4, 2.8, n), 0, 25)

    work_hours_per_week = _clip(rng.normal(46, 8, n), 30, 75)
    sleep_hours = _clip(rng.normal(6.5, 1.1, n) - 0.02 * (work_hours_per_week - 45), 3, 9)
    physical_activity_min_week = _clip(rng.normal(120, 90, n) - 1.5 * (work_hours_per_week - 45), 0, 500)
    diet_score = _clip(rng.normal(6, 1.8, n), 1, 10)
    screen_time_hours = _clip(rng.normal(7, 2, n) + 0.05 * (work_hours_per_week - 45), 1, 16)
    caffeine_intake_mg = _clip(rng.normal(250, 120, n), 0, 800)
    meditation_min_week = _clip(rng.exponential(25, n), 0, 300)
    social_support_score = _clip(rng.normal(6, 2, n), 1, 10)
    job_satisfaction_score = _clip(rng.normal(6.2, 2, n), 1, 10)
    work_life_balance_score = _clip(
        10 - 0.06 * (work_hours_per_week - 40) + rng.normal(0, 1.2, n), 1, 10
    )
    past_medical_leaves = rng.poisson(1.2, n)

    # Latent "true stress" drives stress_level, burnout_risk in a
    # non-trivial, non-linear way so models have real signal to learn.
    latent_stress = (
        0.35 * (work_hours_per_week - 40)
        - 0.9 * (sleep_hours - 7)
        - 0.015 * physical_activity_min_week
        - 0.4 * (work_life_balance_score - 5)
        - 0.3 * (job_satisfaction_score - 5)
        - 0.25 * (social_support_score - 5)
        + 0.02 * screen_time_hours
        + 0.01 * caffeine_intake_mg / 10
        + rng.normal(0, 4, n)
    )
    stress_level = _clip(50 + 2.2 * latent_stress, 0, 100)

    burnout_logit = (
        0.11 * (stress_level - 50)
        - 0.35 * (job_satisfaction_score - 5)
        - 0.25 * (work_life_balance_score - 5)
        - 1.5 * (sleep_hours - 7)
        + rng.normal(0, 1.6, n)
    )
    burnout_prob = 1 / (1 + np.exp(-burnout_logit / 1.6))
    burnout_risk = (rng.random(n) < burnout_prob).astype(int)

    df = pd.DataFrame(
        {
            "employee_id": [f"EMP{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "gender": gender,
            "department": department,
            "tenure_years": np.round(tenure_years, 1),
            "work_hours_per_week": np.round(work_hours_per_week, 1),
            "sleep_hours": np.round(sleep_hours, 2),
            "physical_activity_min_week": np.round(physical_activity_min_week, 1),
            "diet_score": np.round(diet_score, 1),
            "screen_time_hours": np.round(screen_time_hours, 2),
            "caffeine_intake_mg": np.round(caffeine_intake_mg, 1),
            "meditation_min_week": np.round(meditation_min_week, 1),
            "social_support_score": np.round(social_support_score, 1),
            "job_satisfaction_score": np.round(job_satisfaction_score, 1),
            "work_life_balance_score": np.round(work_life_balance_score, 1),
            "past_medical_leaves": past_medical_leaves,
            "stress_level": np.round(stress_level, 1),   # regression target
            "burnout_risk": burnout_risk,                 # classification target
        }
    )

    # Inject a small % of missing values + a few duplicate rows so the
    # preprocessing pipeline has real cleaning work to do (mirrors the
    # "Common Mistakes: data leakage / poor cleaning" warning in the brief).
    for col in ["sleep_hours", "diet_score", "social_support_score", "physical_activity_min_week"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan
    df = pd.concat([df, df.sample(5, random_state=seed)], ignore_index=True)

    return df


def generate_weekly_tracking(cross_sectional: pd.DataFrame, n_weeks=N_WEEKS, seed=RANDOM_SEED) -> pd.DataFrame:
    """
    Builds a per-employee, per-week behavioral trajectory that drifts
    toward (or away from) burnout, giving an LSTM something temporal
    to actually learn beyond what the cross-sectional snapshot shows.
    """
    rng = np.random.default_rng(seed + 1)
    base = cross_sectional.drop_duplicates(subset="employee_id").reset_index(drop=True)

    rows = []
    for _, emp in base.iterrows():
        drift = rng.normal(0, 0.6)  # per-employee trend: worsening (+) or improving (-)
        sleep = emp["sleep_hours"] if pd.notna(emp["sleep_hours"]) else 6.5
        activity = emp["physical_activity_min_week"] if pd.notna(emp["physical_activity_min_week"]) else 100
        stress = emp["stress_level"]

        for week in range(1, n_weeks + 1):
            sleep = _clip(sleep - 0.03 * drift + rng.normal(0, 0.3), 3, 9)
            activity = _clip(activity - 3 * drift + rng.normal(0, 15), 0, 500)
            work_hours = _clip(emp["work_hours_per_week"] + drift * 0.4 + rng.normal(0, 2), 30, 80)
            mood_score = _clip(7 - 0.5 * drift * (week / n_weeks) + rng.normal(0, 0.8), 1, 10)
            stress = _clip(stress + 0.8 * drift + rng.normal(0, 3), 0, 100)

            rows.append(
                {
                    "employee_id": emp["employee_id"],
                    "week": week,
                    "sleep_hours": round(sleep, 2),
                    "physical_activity_min": round(activity, 1),
                    "work_hours": round(work_hours, 1),
                    "mood_score": round(mood_score, 1),
                    "weekly_stress_level": round(stress, 1),
                }
            )

    weekly_df = pd.DataFrame(rows)
    # final label = burnout_risk already defined in cross_sectional (ground truth)
    weekly_df = weekly_df.merge(base[["employee_id", "burnout_risk"]], on="employee_id", how="left")
    return weekly_df


def main():
    cross = generate_cross_sectional()
    weekly = generate_weekly_tracking(cross)

    cross_path = RAW_DIR / "employee_wellness_cross_sectional.csv"
    weekly_path = RAW_DIR / "employee_weekly_tracking.csv"
    cross.to_csv(cross_path, index=False)
    weekly.to_csv(weekly_path, index=False)

    print(f"Wrote {len(cross):,} rows -> {cross_path}")
    print(f"Wrote {len(weekly):,} rows -> {weekly_path}")


if __name__ == "__main__":
    main()
