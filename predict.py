"""
DualSync – Inference Module
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "saved_model"

rf_model = joblib.load(MODEL_DIR / "dualsync_rf_model.joblib")
gb_model = joblib.load(MODEL_DIR / "dualsync_gb_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
FEATURE_COLS = joblib.load(MODEL_DIR / "feature_cols.joblib")
LABEL_MAP = joblib.load(MODEL_DIR / "label_map.joblib")


def _build_feature_vector(data: dict) -> pd.DataFrame:
    row = {col: data.get(col, 0) for col in FEATURE_COLS}
    return pd.DataFrame([row])[FEATURE_COLS]


def predict_risk(user_data: dict, use_ensemble: bool = True) -> dict:
    X = _build_feature_vector(user_data)

    proba_rf = rf_model.predict_proba(X)[0]
    X_scaled = scaler.transform(X)
    proba_gb = gb_model.predict_proba(X_scaled)[0]

    if use_ensemble:
        proba = (proba_rf + proba_gb) / 2
    else:
        proba = proba_rf

    risk_label = int(np.argmax(proba))
    risk_score = float(np.dot(proba, [15, 55, 90]))

    recommendations = _generate_recommendations(user_data, risk_label, proba)
    traffic = {0: "green", 1: "yellow", 2: "red"}[risk_label]

    return {
        "risk_label": risk_label,
        "risk_text": LABEL_MAP[risk_label],
        "risk_score": round(risk_score, 1),
        "probabilities": {
            "Low": round(float(proba[0]), 3),
            "Medium": round(float(proba[1]), 3),
            "High": round(float(proba[2]), 3),
        },
        "traffic_light": traffic,
        "recommendations": recommendations,
        "model_version": "dualsync_v1_rf_gb_ensemble",
        "timestamp": pd.Timestamp.now().isoformat(),
    }


def _generate_recommendations(data: dict, risk_label: int, proba: np.ndarray) -> list:
    recs = []

    sleep = data.get("sleep_hours", 7)
    hrv = data.get("hrv_score", 55)
    energy = data.get("subjective_energy", 6)
    exams = data.get("exam_count_3d", 0)
    train_h = data.get("training_hours_today", 0)
    intensity = data.get("training_intensity", 2)
    hydration = data.get("hydration_adherence", 1)
    nutrition = data.get("nutrition_adherence", 1)
    days_rest = data.get("days_since_rest", 2)

    if risk_label == 2:
        recs.append("High dual-load risk detected. Prioritize recovery today.")
        if sleep < 7:
            recs.append(f"You only slept {sleep:.1f}h – aim for at least 7.5h tonight.")
        if hrv < 45:
            recs.append("HRV is low – skip high-intensity training and do active recovery only.")
        if exams >= 2:
            recs.append("Multiple exams ahead – protect a 90-min deep-work block and move non-essential training.")
        if intensity >= 4 and train_h > 1.5:
            recs.append("High intensity + high volume today – reduce training duration by 30-40%.")
        if hydration == 0:
            recs.append("Hydration is low – drink 500 ml water in the next 30 minutes.")
        if nutrition == 0:
            recs.append("Nutrition adherence is poor – prioritize a balanced meal within 1 hour.")
        if days_rest >= 5:
            recs.append("No full rest day in 5+ days – schedule a complete rest or very light session tomorrow.")

    elif risk_label == 1:
        recs.append("Moderate dual-load. Stay mindful of recovery.")
        if sleep < 6.5:
            recs.append("Sleep is on the low side – try to add 45-60 min tonight.")
        if energy <= 4:
            recs.append("Energy is low – choose light study or technique work instead of heavy sessions.")
        if exams >= 1 and train_h > 2:
            recs.append("Exam + training load – keep training but reduce intensity to Zone 2.")
        if hydration == 0:
            recs.append("Remember to hydrate – set a reminder every 90 minutes.")

    else:
        recs.append("Dual-load is currently balanced. Good job!")
        if energy >= 8 and hrv > 60:
            recs.append("You have good recovery capacity – this is a solid day for a harder session or deep study.")
        if exams == 0 and train_h < 1:
            recs.append("Light day – consider adding a short skill or mobility session if you feel good.")

    if len(recs) < 3:
        recs.append("Log how you feel after today’s sessions so the model can personalize further.")

    return recs[:5]


if __name__ == "__main__":
    example_high = {
        "exam_count_3d": 2, "deadline_count_7d": 4, "study_hours_today": 5.5,
        "calendar_density": 8.0, "training_hours_today": 2.5, "training_intensity": 4,
        "recent_load_7d": 14, "sleep_hours": 5.5, "hrv_score": 38, "resting_hr": 72,
        "hydration_adherence": 0, "nutrition_adherence": 0, "subjective_energy": 3,
        "avg_past_risk": 0.65, "days_since_rest": 6,
    }
    result = predict_risk(example_high)
    print(f"Risk: {result['risk_text']} ({result['risk_score']}/100)")
    print("Recommendations:")
    for r in result["recommendations"]:
        print(f"  • {r}")
