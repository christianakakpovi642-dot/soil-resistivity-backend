import logging
import os
import sys
import threading

import joblib
import numpy as np

# Import nécessaire même si non utilisé directement ici :
# joblib.load() a besoin que la classe HybridRegressor soit importable
# depuis le même chemin de module que lors de la sauvegarde du .pkl.
from app.hybrid_model import HybridRegressor
from app.config import MODEL_PATH, SCALER_PATH

logger = logging.getLogger("model_service")


def _register_main_module_compat():
    """
    Correctif de compatibilité pickle.

    Si ton .pkl a été généré en exécutant le script d'entraînement directement
    (`python script.py`), Python a enregistré HybridRegressor comme appartenant
    au module `__main__`. Ici, la classe vit dans `app.hybrid_model`, donc le
    dépickling échoue avec `AttributeError: module '__main__' has no attribute
    'HybridRegressor'`.

    On corrige ça en exposant la classe sous ce nom, quel que soit le module
    réellement utilisé comme point d'entrée (uvicorn, pytest, etc.).
    """
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "HybridRegressor"):
        main_module.HybridRegressor = HybridRegressor


class ModelService:
    """
    Charge le modèle au démarrage et sert les prédictions.

    Thread-safe : un verrou protège les lectures (predict) et l'écriture
    (swap_model, appelé après un ré-entraînement) pour qu'une requête ne
    tombe jamais sur un modèle à moitié remplacé.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self.model = None
        self.scaler = None
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Modèle introuvable à {MODEL_PATH}. "
                "Copie ton fichier modele_hybride_wenner.pkl dans le dossier 'models/'."
            )
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(
                f"Scaler introuvable à {SCALER_PATH}. "
                "Copie ton fichier scaler_wenner.pkl dans le dossier 'models/'."
            )

        _register_main_module_compat()

        with self._lock:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        logger.info("Modèle et scaler chargés avec succès depuis %s", MODEL_PATH)

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self.model is not None and self.scaler is not None

    def swap_model(self, model, scaler):
        """Remplace le modèle en production par une version fraîchement ré-entraînée."""
        with self._lock:
            self.model = model
            self.scaler = scaler
        logger.info("Nouveau modèle publié en production")

    def predict_one(self, latitude: float, longitude: float, altitude: float) -> float:
        with self._lock:
            model, scaler = self.model, self.scaler
        X = np.array([[latitude, longitude, altitude]])
        Xs = scaler.transform(X)
        prediction = model.predict(Xs)
        return float(prediction[0])

    def predict_batch(self, points: list[tuple[float, float, float]]) -> list[float]:
        with self._lock:
            model, scaler = self.model, self.scaler
        X = np.array(points)
        Xs = scaler.transform(X)
        predictions = model.predict(Xs)
        return [float(p) for p in predictions]


# Instance unique partagée par toute l'application (chargée au démarrage)
model_service = ModelService()
