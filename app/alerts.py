"""
Alertes de maintenance : détecte les situations qui méritent l'attention
d'un humain (performance en baisse, ré-entraînement en échec, modèle qui
n'a pas été mis à jour depuis longtemps, erreurs répétées côté serveur),
les stocke, et envoie un email pour les plus critiques.

Chaque type d'alerte est dédupliqué : tant qu'une alerte d'un type donné
n'est pas résolue, une nouvelle occurrence du même problème ne recrée pas
de doublon — elle est simplement ignorée côté stockage (mais reste loguée).
Quand la situation redevient normale, l'alerte correspondante est résolue
automatiquement.
"""

import logging
from datetime import datetime

from app import database
from app.email_service import send_email
from app.config import ALERT_EMAIL_MIN_SEVERITY, MODEL_STALE_DAYS, MAX_CONSECUTIVE_PREDICT_ERRORS

logger = logging.getLogger("alerts")

SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}

# Types d'alerte
TYPE_RETRAIN_FAILED = "RETRAIN_FAILED"
TYPE_MODEL_PERFORMANCE_LOW = "MODEL_PERFORMANCE_LOW"
TYPE_MODEL_STALE = "MODEL_STALE"
TYPE_SERVER_HEALTH = "SERVER_HEALTH"

# Compteur d'erreurs consécutives sur /predict.
# Vit en mémoire du process : suffisant pour un déploiement à un seul worker
# (voir la remarque sur --workers 1 dans le README).
_consecutive_predict_errors = 0


def _should_email(severity: str) -> bool:
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(ALERT_EMAIL_MIN_SEVERITY, 2)


def trigger_alert(alert_type: str, severity: str, message: str):
    """Crée l'alerte (si pas déjà active) et envoie un email si la sévérité le justifie."""
    alert = database.create_alert(alert_type, severity, message)
    if alert is None:
        logger.info("Alerte [%s] déjà active, pas de doublon créé", alert_type)
        return

    logger.warning("ALERTE [%s] %s: %s", severity.upper(), alert_type, message)

    if _should_email(severity):
        send_email(
            subject=f"[Soil Resistivity API] Alerte {severity.upper()} — {alert_type}",
            body=message,
        )


def resolve(alert_type: str):
    database.resolve_alerts_of_type(alert_type)


# ================================
# Santé du serveur (erreurs sur /predict)
# ================================

def record_predict_success():
    """A appeler après une prédiction réussie : réinitialise le compteur d'erreurs."""
    global _consecutive_predict_errors
    if _consecutive_predict_errors > 0:
        resolve(TYPE_SERVER_HEALTH)
    _consecutive_predict_errors = 0


def record_predict_error(error_message: str):
    """A appeler après une prédiction en échec : incrémente le compteur, alerte si seuil atteint."""
    global _consecutive_predict_errors
    _consecutive_predict_errors += 1
    if _consecutive_predict_errors >= MAX_CONSECUTIVE_PREDICT_ERRORS:
        trigger_alert(
            TYPE_SERVER_HEALTH,
            "critical",
            f"{_consecutive_predict_errors} erreurs consécutives sur /predict. "
            f"Dernière erreur : {error_message}",
        )


# ================================
# Fraîcheur du modèle
# ================================

def check_model_staleness():
    """Alerte si le modèle n'a pas été ré-entraîné depuis longtemps alors que des données attendent."""
    stats = database.get_stats()
    pending = stats["history_pending_retrain"]
    last_retrain = stats["last_retrain"]

    if pending == 0:
        # Rien de nouveau à apprendre : pas la peine d'alerter sur l'ancienneté.
        return

    if last_retrain is None:
        return  # jamais ré-entraîné depuis le lancement : pas assez d'info pour juger

    try:
        reference_date = datetime.strptime(last_retrain["timestamp"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return

    age_days = (datetime.utcnow() - reference_date).days
    if age_days >= MODEL_STALE_DAYS:
        trigger_alert(
            TYPE_MODEL_STALE,
            "warning",
            f"Le modèle n'a pas été ré-entraîné depuis {age_days} jours, alors que "
            f"{pending} nouveaux points attendent d'être intégrés.",
        )
    else:
        resolve(TYPE_MODEL_STALE)
