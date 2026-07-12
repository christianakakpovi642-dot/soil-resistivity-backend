"""
Ré-entraînement automatique du modèle hybride, déclenché par l'accumulation
de nouvelles prédictions/mesures envoyées depuis l'app.

Logique reprise de ton script original (reentrainement_automatique) :
  1. Déclenchement si le nombre de nouveaux points >= RETRAIN_THRESHOLD
  2. OU si la performance (R2) du modèle actuel descend sous PERFORMANCE_R2_THRESHOLD
  3. Ré-entraînement sur (données de seed + historique complet)
  4. Le nouveau modèle est sauvegardé sur disque ET publié en mémoire
     (aucun redémarrage du serveur n'est nécessaire)

Différence volontaire avec le script original : celui-ci vidait l'historique
CSV après chaque ré-entraînement sans le refondre dans les données initiales,
ce qui faisait "oublier" les anciens points à chaque nouveau cycle. Ici,
l'historique est conservé en base et systématiquement recombiné avec les
données de seed à chaque ré-entraînement — le modèle ne perd jamais de
données déjà collectées.
"""

import logging
import time
from threading import Lock

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.hybrid_model import HybridRegressor
from app.config import (
    RETRAIN_THRESHOLD,
    PERFORMANCE_R2_THRESHOLD,
    MODEL_PATH,
    SCALER_PATH,
    RF_WEIGHT,
    KNN_WEIGHT,
)
from app import database
from app import alerts
from app.model_service import model_service

logger = logging.getLogger("retrain_service")

# Empêche deux ré-entraînements de tourner en parallèle
_retrain_lock = Lock()


def _evaluate_current_model(df: pd.DataFrame):
    """Évalue le modèle actuellement en service sur le jeu de données fourni."""
    if not model_service.is_loaded or df.empty:
        return None
    try:
        X = df[["latitude", "longitude", "altitude"]].to_numpy()
        y = df["wenner"].to_numpy()
        Xs = model_service.scaler.transform(X)
        y_pred = model_service.model.predict(Xs)
        return r2_score(y, y_pred)
    except Exception as e:
        logger.warning("Impossible d'évaluer le modèle actuel: %s", e)
        return None


def should_retrain() -> tuple[bool, str]:
    """Détermine si un ré-entraînement est nécessaire, et pourquoi."""
    n_new = database.count_unused_history()
    if n_new >= RETRAIN_THRESHOLD:
        return True, f"{n_new} nouveaux points >= seuil ({RETRAIN_THRESHOLD})"

    df = database.get_training_dataframe()
    if len(df) < 10:
        # Pas assez de données pour qu'une évaluation de performance ait un
        # sens statistique — inutile de déclencher une tentative de
        # ré-entraînement qui échouera de toute façon faute de données.
        return False, ""

    r2 = _evaluate_current_model(df)
    if r2 is not None and r2 < PERFORMANCE_R2_THRESHOLD:
        return True, f"Performance R2 ({r2:.3f}) < seuil ({PERFORMANCE_R2_THRESHOLD})"

    return False, ""


def perform_retrain(reason: str = "manuel", blocking: bool = False) -> dict | None:
    """
    Ré-entraîne le modèle hybride sur (seed + historique complet) et le publie.

    `blocking` : si False (déclenchement automatique en tâche de fond), abandonne
    immédiatement si un ré-entraînement est déjà en cours. Si True (déclenchement
    manuel via /admin/retrain), attend jusqu'à 30s que le verrou se libère plutôt
    que d'échouer immédiatement en cas de contention passagère.

    Retourne un résumé des métriques, ou None si le ré-entraînement n'a pas eu lieu.
    """
    acquired = _retrain_lock.acquire(blocking=blocking, timeout=30 if blocking else -1)
    if not acquired:
        logger.info("Ré-entraînement déjà en cours, requête ignorée")
        return None

    try:
        logger.info("Début du ré-entraînement (raison: %s)", reason)
        start = time.time()

        df = database.get_training_dataframe()
        if len(df) < 10:
            logger.warning("Pas assez de données pour ré-entraîner (%d points, minimum 10)", len(df))
            return None

        X = df[["latitude", "longitude", "altitude"]].to_numpy()
        y = df["wenner"].to_numpy()

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Mêmes hyperparamètres que ceux retenus par ton script après optimisation.
        # Pas de nouvelle recherche sur grille ici : GridSearchCV est trop coûteux
        # pour tourner à chaque ré-entraînement déclenché en tâche de fond.
        # knn_n_neighbors est plafonné à (n_échantillons - 1) : au tout début de
        # la collecte de données, il peut y avoir moins de 11 points disponibles.
        knn_n_neighbors = min(11, len(df) - 1)
        model = HybridRegressor(
            rf_weight=RF_WEIGHT,
            knn_weight=KNN_WEIGHT,
            rf_n_estimators=100,
            rf_max_depth=10,
            rf_min_samples_split=min(10, len(df)),
            rf_min_samples_leaf=min(4, max(1, len(df) // 5)),
            knn_n_neighbors=knn_n_neighbors,
            knn_weights="uniform",
        )
        model.fit(Xs, y)

        y_pred = model.predict(Xs)
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))

        # Sauvegarde sur disque (persistance) puis publication en mémoire (effet immédiat)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        model_service.swap_model(model, scaler)

        database.mark_history_as_used()
        duration = time.time() - start
        database.log_retrain(
            n_points_total=len(df),
            n_new_points_after=database.count_unused_history(),
            r2_train=r2,
            mae_train=mae,
            rmse_train=rmse,
            duration_s=duration,
            reason=reason,
        )

        logger.info(
            "Ré-entraînement terminé en %.1fs — %d points, R2=%.4f, MAE=%.2f Ohm.m",
            duration, len(df), r2, mae,
        )

        # Le ré-entraînement a réussi : ce n'est plus un problème s'il y en avait un.
        alerts.resolve(alerts.TYPE_RETRAIN_FAILED)
        alerts.resolve(alerts.TYPE_MODEL_STALE)

        if r2 < PERFORMANCE_R2_THRESHOLD:
            alerts.trigger_alert(
                alerts.TYPE_MODEL_PERFORMANCE_LOW,
                "warning",
                f"Le modèle ré-entraîné a un R2 de {r2:.3f} sur ses propres données "
                f"d'entraînement, en dessous du seuil attendu ({PERFORMANCE_R2_THRESHOLD}). "
                f"Vérifie la qualité des données collectées récemment.",
            )
        else:
            alerts.resolve(alerts.TYPE_MODEL_PERFORMANCE_LOW)

        return {
            "n_points": len(df),
            "r2_train": round(r2, 4),
            "mae_train": round(mae, 2),
            "rmse_train": round(rmse, 2),
            "duration_s": round(duration, 1),
            "reason": reason,
        }
    except Exception as e:
        logger.error("Erreur lors du ré-entraînement: %s", e)
        alerts.trigger_alert(
            alerts.TYPE_RETRAIN_FAILED,
            "critical",
            f"Le ré-entraînement automatique a échoué avec l'erreur : {e}",
        )
        return None
    finally:
        _retrain_lock.release()


def maybe_retrain():
    """
    A appeler en tâche de fond après une prédiction : vérifie les conditions
    de ré-entraînement et le déclenche si nécessaire. Vérifie aussi la
    fraîcheur du modèle. Ne bloque jamais la réponse déjà envoyée au client.
    """
    try:
        needed, reason = should_retrain()
        if needed:
            perform_retrain(reason=reason)
        else:
            alerts.check_model_staleness()
    except Exception as e:
        logger.error("Erreur lors de la vérification du ré-entraînement: %s", e)
