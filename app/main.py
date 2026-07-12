import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    RetrainResult,
    StatsResponse,
    AlertResponse,
    AppConfigUpdate,
    AppConfigResponse,
    VersionCheckRequest,
    VersionCheckResponse,
)
from app.model_service import model_service
from app import database
from app import retrain_service
from app import alerts
from app.version_utils import compare_versions
from app.config import ADMIN_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")

app = FastAPI(
    title="Soil Resistivity API",
    description="API de prédiction de résistivité du sol (méthode Wenner) avec ré-entraînement automatique",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    database.init_db()
    logger.info("Base de données initialisée")


def _check_admin_key(x_admin_key: str | None):
    """Protection légère des endpoints admin. Désactivée si ADMIN_API_KEY n'est pas définie."""
    if ADMIN_API_KEY and x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Clé admin invalide ou manquante")


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Vérifie que l'API tourne et que le modèle est bien chargé."""
    return HealthResponse(status="ok", model_loaded=model_service.is_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Prédit la résistivité du sol pour un point GPS unique.

    Si `measured_resistivity` est fourni (vraie mesure de terrain), elle est
    stockée comme donnée d'entraînement fiable. Sinon, la prédiction du modèle
    est stockée comme donnée d'apprentissage (comportement du script original).

    Après la réponse, une vérification de ré-entraînement est lancée en tâche
    de fond : elle ne ralentit jamais cette requête.
    """
    try:
        resistivity = model_service.predict_one(
            latitude=request.latitude,
            longitude=request.longitude,
            altitude=request.altitude,
        )

        if request.measured_resistivity is not None:
            label_value = request.measured_resistivity
            source = "measured"
        else:
            label_value = resistivity
            source = "predicted"

        database.add_prediction(
            latitude=request.latitude,
            longitude=request.longitude,
            altitude=request.altitude,
            wenner=label_value,
            source=source,
        )

        background_tasks.add_task(retrain_service.maybe_retrain)
        alerts.record_predict_success()

        return PredictionResponse(
            latitude=request.latitude,
            longitude=request.longitude,
            altitude=request.altitude,
            resistivity=round(resistivity, 2),
            source=source,
        )
    except Exception as e:
        logger.error("Erreur de prédiction: %s", e)
        alerts.record_predict_error(str(e))
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
    """Prédit la résistivité du sol pour plusieurs points GPS en une seule requête."""
    try:
        coords = [(p.latitude, p.longitude, p.altitude) for p in request.points]
        predictions = model_service.predict_batch(coords)

        results = []
        for p, pred in zip(request.points, predictions):
            if p.measured_resistivity is not None:
                label_value, source = p.measured_resistivity, "measured"
            else:
                label_value, source = pred, "predicted"

            database.add_prediction(
                latitude=p.latitude,
                longitude=p.longitude,
                altitude=p.altitude,
                wenner=label_value,
                source=source,
            )

            results.append(
                PredictionResponse(
                    latitude=p.latitude,
                    longitude=p.longitude,
                    altitude=p.altitude,
                    resistivity=round(pred, 2),
                    source=source,
                )
            )

        background_tasks.add_task(retrain_service.maybe_retrain)

        return BatchPredictionResponse(results=results)
    except Exception as e:
        logger.error("Erreur de prédiction batch: %s", e)
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {e}")


# ================================
# Endpoints d'administration
# ================================

@app.get("/admin/stats", response_model=StatsResponse)
def admin_stats(x_admin_key: str | None = Header(default=None)):
    """Statistiques sur l'historique collecté et le dernier ré-entraînement."""
    _check_admin_key(x_admin_key)
    return StatsResponse(**database.get_stats())


@app.post("/admin/retrain", response_model=RetrainResult)
def admin_force_retrain(x_admin_key: str | None = Header(default=None)):
    """Force un ré-entraînement immédiat (bloquant), quel que soit le seuil."""
    _check_admin_key(x_admin_key)
    result = retrain_service.perform_retrain(reason="déclenché manuellement", blocking=True)
    if result is None:
        raise HTTPException(
            status_code=409,
            detail="Ré-entraînement non effectué (déjà en cours, ou pas assez de données)",
        )
    return RetrainResult(**result)


# ================================
# Alertes de maintenance
# ================================

@app.get("/admin/alerts", response_model=list[AlertResponse])
def admin_list_alerts(
    resolved: bool | None = None,
    severity: str | None = None,
    x_admin_key: str | None = Header(default=None),
):
    """Liste les alertes de maintenance. Filtre optionnel par statut/sévérité."""
    _check_admin_key(x_admin_key)
    rows = database.get_alerts(resolved=resolved, severity=severity)
    return [
        AlertResponse(
            id=r["id"],
            type=r["type"],
            severity=r["severity"],
            message=r["message"],
            resolved=bool(r["resolved"]),
            created_at=r["created_at"],
            resolved_at=r["resolved_at"],
        )
        for r in rows
    ]


@app.post("/admin/alerts/{alert_id}/resolve")
def admin_resolve_alert(alert_id: int, x_admin_key: str | None = Header(default=None)):
    """Marque une alerte comme résolue manuellement."""
    _check_admin_key(x_admin_key)
    updated = database.resolve_alert(alert_id)
    if not updated:
        raise HTTPException(status_code=404, detail="Alerte introuvable ou déjà résolue")
    return {"status": "resolved", "alert_id": alert_id}


# ================================
# Version de l'application mobile
# ================================

@app.get("/admin/app-config", response_model=AppConfigResponse)
def admin_get_app_config(x_admin_key: str | None = Header(default=None)):
    """Consulte la configuration de version actuellement publiée."""
    _check_admin_key(x_admin_key)
    return AppConfigResponse(**database.get_app_config())


@app.put("/admin/app-config", response_model=AppConfigResponse)
def admin_update_app_config(
    update: AppConfigUpdate, x_admin_key: str | None = Header(default=None)
):
    """
    Met à jour la version publiée de l'app (et/ou l'URL/message associés),
    sans avoir besoin de redéployer le backend.
    """
    _check_admin_key(x_admin_key)
    database.set_app_config(update.model_dump(exclude_none=True))
    return AppConfigResponse(**database.get_app_config())


@app.post("/app/check-version", response_model=VersionCheckResponse)
def check_app_version(request: VersionCheckRequest):
    """
    Endpoint public appelé par l'app au démarrage pour savoir si une mise à
    jour est disponible ou obligatoire.
    """
    config = database.get_app_config()
    latest = config.get("latest_version", "1.0.0")
    min_supported = config.get("min_supported_version", "1.0.0")
    update_url = config.get("update_url", "")
    message = config.get("update_message", "")

    update_available = compare_versions(request.current_version, latest) < 0
    force_update = compare_versions(request.current_version, min_supported) < 0

    return VersionCheckResponse(
        update_available=update_available,
        force_update=force_update,
        latest_version=latest,
        update_url=update_url,
        message=message if (update_available or force_update) else "",
    )
