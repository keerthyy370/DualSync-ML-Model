from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import predict_risk

app = FastAPI(
    title="DualSync Dual-Load Prediction API",
    description="Personalized overload risk prediction for student-athletes",
    version="1.0.0"
)


class UserFeatures(BaseModel):
    exam_count_3d: float = Field(0, ge=0)
    deadline_count_7d: float = Field(0, ge=0)
    study_hours_today: float = Field(0, ge=0, le=16)
    calendar_density: float = Field(0, ge=0, le=10)
    training_hours_today: float = Field(0, ge=0, le=8)
    training_intensity: int = Field(2, ge=1, le=5)
    recent_load_7d: float = Field(0, ge=0)
    sleep_hours: float = Field(7.0, ge=0, le=12)
    hrv_score: float = Field(55, ge=10, le=120)
    resting_hr: float = Field(60, ge=40, le=120)
    hydration_adherence: int = Field(1, ge=0, le=1)
    nutrition_adherence: int = Field(1, ge=0, le=1)
    subjective_energy: int = Field(6, ge=1, le=10)
    avg_past_risk: float = Field(0.4, ge=0, le=1)
    days_since_rest: int = Field(2, ge=0, le=14)


@app.get("/")
def root():
    return {
        "message": "DualSync Dual-Load Prediction API is running",
        "endpoints": {
            "POST /predict": "Get personalized risk + recommendations",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "dualsync_v1_rf_gb_ensemble"}


@app.post("/predict")
def predict(features: UserFeatures):
    try:
        result = predict_risk(features.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))