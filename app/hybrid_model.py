"""
Définition de la classe HybridRegressor.

IMPORTANT : joblib.load() a besoin de retrouver la définition EXACTE de cette
classe (même nom, même module de référence) pour pouvoir désérialiser le
modèle .pkl. Ce fichier doit rester cohérent avec celui utilisé lors de
l'entraînement (RandomForest + KNN pondérés).
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class HybridRegressor(BaseEstimator, RegressorMixin):
    """Modèle hybride combinant Random Forest et KNN."""

    def __init__(self, rf_weight=0.6, knn_weight=0.4,
                 rf_n_estimators=200, rf_max_depth=20, rf_min_samples_split=5,
                 rf_min_samples_leaf=2, knn_n_neighbors=5, knn_weights='distance'):
        self.rf_weight = rf_weight
        self.knn_weight = knn_weight
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.knn_n_neighbors = knn_n_neighbors
        self.knn_weights = knn_weights

        self.rf_model = None
        self.knn_model = None
        self.is_fitted = False

    def fit(self, X, y):
        self.rf_model = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split,
            min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X, y)

        self.knn_model = KNeighborsRegressor(
            n_neighbors=self.knn_n_neighbors,
            weights=self.knn_weights,
            n_jobs=-1
        )
        self.knn_model.fit(X, y)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas encore été entraîné")

        rf_pred = self.rf_model.predict(X)
        knn_pred = self.knn_model.predict(X)
        return self.rf_weight * rf_pred + self.knn_weight * knn_pred

    def get_params(self, deep=True):
        return {
            'rf_weight': self.rf_weight,
            'knn_weight': self.knn_weight,
            'rf_n_estimators': self.rf_n_estimators,
            'rf_max_depth': self.rf_max_depth,
            'rf_min_samples_split': self.rf_min_samples_split,
            'rf_min_samples_leaf': self.rf_min_samples_leaf,
            'knn_n_neighbors': self.knn_n_neighbors,
            'knn_weights': self.knn_weights
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_feature_importances(self):
        if not self.is_fitted:
            return None
        rf_importances = self.rf_model.feature_importances_
        knn_importances = np.ones_like(rf_importances) / len(rf_importances)
        return self.rf_weight * rf_importances + self.knn_weight * knn_importances
