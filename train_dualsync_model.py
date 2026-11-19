"""
DualSync – Dual-Load Risk Prediction Model
Train a Random Forest + Gradient Boosting ensemble.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_synthetic_data(n_samples=2500, random_state=42):
    np.random.seed(random_state)

    exam_count_3d = np.random.poisson(1.2, n_samples)
    deadline_count_7d = np.random.poisson(2.5, n_samples)
    study_hours_today = np.clip(np.random.normal(3.5, 2.0, n_samples), 0, 12)
    calendar_density = np.clip(np.random.beta(2, 5, n_samples) * 10, 0, 10)

    training_hours_today = np.clip(np.random.normal(1.8, 1.2, n_samples), 0, 6)
    training_intensity = np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.25, 0.30, 0.20, 0.10])
    recent_load_7d = np.clip(np.random.normal(8, 4, n_samples), 0, 25)

    sleep_hours = np.clip(np.random.normal(6.8, 1.3, n_samples), 3, 10)
    hrv_score = np.clip(np.random.normal(55, 15, n_samples), 20, 100)
    resting_hr = np.clip(np.random.normal(62, 8, n_samples), 45, 95)
    hydration_adherence = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    nutrition_adherence = np.random.choice([0, 1], n_samples, p=[0.40, 0.60])
    subjective_energy = np.random.randint(1, 11, n_samples)

    avg_past_risk = np.clip(np.random.normal(0.45, 0.18, n_samples), 0.1, 0.9)
    days_since_rest = np.random.randint(0, 8, n_samples)

    df = pd.DataFrame({
        "exam_count_3d": exam_count_3d,
        "deadline_count_7d": deadline_count_7d,
        "study_hours_today": study_hours_today,
        "calendar_density": calendar_density,
        "training_hours_today": training_hours_today,
        "training_intensity": training_intensity,
        "recent_load_7d": recent_load_7d,
        "sleep_hours": sleep_hours,
        "hrv_score": hrv_score,
        "resting_hr": resting_hr,
        "hydration_adherence": hydration_adherence,
        "nutrition_adherence": nutrition_adherence,
        "subjective_energy": subjective_energy,
        "avg_past_risk": avg_past_risk,
        "days_since_rest": days_since_rest,
    })

    academic_pressure = (
        0.35 * exam_count_3d +
        0.15 * deadline_count_7d +
        0.20 * study_hours_today +
        0.25 * calendar_density
    )
    athletic_pressure = (
        0.40 * training_hours_today +
        0.30 * training_intensity +
        0.25 * recent_load_7d / 5
    )
    recovery_deficit = (
        (8 - sleep_hours) * 0.8 +
        (70 - hrv_score) * 0.15 +
        (resting_hr - 55) * 0.1 +
        (1 - hydration_adherence) * 2.5 +
        (1 - nutrition_adherence) * 2.0 +
        (6 - subjective_energy) * 0.4 +
        days_since_rest * 0.4
    )

    risk_continuous = (
        0.35 * academic_pressure +
        0.30 * athletic_pressure +
        0.35 * recovery_deficit +
        0.10 * avg_past_risk * 8
    )
    risk_continuous += np.random.normal(0, 0.25, n_samples)

    risk_label = np.zeros(n_samples, dtype=int)
    risk_label[risk_continuous > 3.85] = 1
    risk_label[risk_continuous > 5.15] = 2

    df["risk_label"] = risk_label
    return df


FEATURE_COLS = [
    "exam_count_3d", "deadline_count_7d", "study_hours_today", "calendar_density",
    "training_hours_today", "training_intensity", "recent_load_7d",
    "sleep_hours", "hrv_score", "resting_hr",
    "hydration_adherence", "nutrition_adherence", "subjective_energy",
    "avg_past_risk", "days_since_rest"
]

TARGET = "risk_label"
LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}


def train_and_save():
    print("=" * 60)
    print("DualSync Dual-Load Risk Model – Training")
    print("=" * 60)

    df = generate_synthetic_data(n_samples=2500)
    print(f"\nGenerated {len(df)} synthetic samples")
    print("Risk distribution:")
    print(df[TARGET].value_counts(normalize=True).sort_index().map(lambda x: f"{x:.1%}"))

    X = df[FEATURE_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.85,
        random_state=42
    )
    gb.fit(X_train_scaled, y_train)

    print("\n" + "-" * 40)
    print("Random Forest Results")
    print("-" * 40)
    y_pred_rf = rf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(classification_report(y_test, y_pred_rf, labels=[0,1,2], target_names=["Low", "Medium", "High"], zero_division=0))

    print("\n" + "-" * 40)
    print("Gradient Boosting Results")
    print("-" * 40)
    y_pred_gb = gb.predict(X_test_scaled)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.3f}")
    print(classification_report(y_test, y_pred_gb, labels=[0,1,2], target_names=["Low", "Medium", "High"], zero_division=0))

    print("\nTop Feature Importances (Random Forest):")
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(importances.head(10).to_string())

    os.makedirs("saved_model", exist_ok=True)
    joblib.dump(rf, "saved_model/dualsync_rf_model.joblib")
    joblib.dump(gb, "saved_model/dualsync_gb_model.joblib")
    joblib.dump(scaler, "saved_model/scaler.joblib")
    joblib.dump(FEATURE_COLS, "saved_model/feature_cols.joblib")
    joblib.dump(LABEL_MAP, "saved_model/label_map.joblib")

    print("\n" + "=" * 60)
    print("Models saved successfully to ./saved_model/")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save()