from typing import Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude GPS")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude GPS")
    altitude: float = Field(..., ge=-500, le=9000, description="Altitude en mètres")
    measured_resistivity: Optional[float] = Field(
        None,
        gt=0,
        description=(
            "Si l'utilisateur a effectué une vraie mesure Wenner sur le terrain, "
            "envoie-la ici. Elle sera utilisée comme donnée d'entraînement fiable. "
            "Si absent, la prédiction du modèle sera utilisée comme donnée d'apprentissage "
            "(comme dans le script original)."
        ),
    )

    class Config:
        json_schema_extra = {
            "example": {
                "latitude": 6.1319,
                "longitude": 1.2228,
                "altitude": 50.0,
                "measured_resistivity": None,
            }
        }


class PredictionResponse(BaseModel):
    latitude: float
    longitude: float
    altitude: float
    resistivity: float
    unit: str = "Ohm.m"
    source: str = "predicted"  # "predicted" ou "measured" selon ce qui a été stocké pour l'entraînement


class BatchPredictionRequest(BaseModel):
    points: list[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class RetrainResult(BaseModel):
    n_points: int
    r2_train: float
    mae_train: float
    rmse_train: float
    duration_s: float
    reason: str


class StatsResponse(BaseModel):
    seed_points: int
    history_total: int
    history_measured: int
    history_predicted: int
    history_pending_retrain: int
    last_retrain: Optional[dict] = None


class AlertResponse(BaseModel):
    id: int
    type: str
    severity: str
    message: str
    resolved: bool
    created_at: str
    resolved_at: Optional[str] = None


class AppConfigUpdate(BaseModel):
    latest_version: Optional[str] = None
    min_supported_version: Optional[str] = None
    update_url: Optional[str] = None
    update_message: Optional[str] = None


class AppConfigResponse(BaseModel):
    latest_version: str
    min_supported_version: str
    update_url: str
    update_message: str


class VersionCheckRequest(BaseModel):
    current_version: str = Field(..., description="Version actuelle de l'app, ex: '1.2.0'")
    platform: str = Field(default="android", description="'android' ou 'ios'")


class VersionCheckResponse(BaseModel):
    update_available: bool
    force_update: bool
    latest_version: str
    update_url: str
    message: str
