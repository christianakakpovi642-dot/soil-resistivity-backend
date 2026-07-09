"""
API de prédiction de résistivité du sol - Méthode Wenner
Modèle hybride: Random Forest + KNN avec validation croisée
"""

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import uvicorn
import os
from datetime import datetime
import io

# ================================
# MODÈLES PYDANTIC
# ================================

class PointCoordonnees(BaseModel):
    """Modèle pour un point de coordonnées"""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude en degrés (-90 à 90)")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude en degrés (-180 à 180)")
    altitude: float = Field(..., ge=-500, le=9000, description="Altitude en mètres (-500 à 9000)")
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if not -90 <= v <= 90:
            raise ValueError('La latitude doit être comprise entre -90 et 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if not -180 <= v <= 180:
            raise ValueError('La longitude doit être comprise entre -180 et 180')
        return v
    
    @validator('altitude')
    def validate_altitude(cls, v):
        if not -500 <= v <= 9000:
            raise ValueError('L\'altitude doit être comprise entre -500 et 9000 mètres')
        return v

class PredictionRequest(BaseModel):
    """Requête de prédiction"""
    points: List[PointCoordonnees]
    
    @validator('points')
    def validate_points_not_empty(cls, v):
        if not v:
            raise ValueError('La liste des points ne peut pas être vide')
        return v

class PredictionResult(BaseModel):
    """Résultat de prédiction pour un point"""
    latitude: float
    longitude: float
    altitude: float
    resistivite_wenner: float = Field(..., description="Résistivité en Ohm.m")

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    predictions: List[PredictionResult]
    nombre_points: int
    resistivite_moyenne: float
    resistivite_min: float
    resistivite_max: float
    timestamp: str

class ModelInfo(BaseModel):
    """Informations sur le modèle"""
    type_modele: str
    poids_rf: float
    poids_knn: float
    rf_n_estimators: int
    rf_max_depth: int
    knn_n_neighbors: int
    knn_weights: str
    est_entraine: bool
    date_chargement: str

class ReentrainementResponse(BaseModel):
    """Réponse pour le réentraînement"""
    message: str
    nombre_points: int
    performance_r2: float
    performance_mae: float
    timestamp: str

# ================================
# INITIALISATION DE L'API
# ================================

app = FastAPI(
    title="API de Prédiction de Résistivité du Sol",
    description="API pour la prédiction de la résistivité du sol selon la méthode Wenner",
    version="1.0.0"
)

# Variables globales pour le modèle
model = None
scaler = None
X_train_global = None
X_test_global = None
y_train_global = None
y_test_global = None

# Constantes
FICHIER_MODELE = "modele_hybride_wenner.pkl"
FICHIER_SCALER = "scaler_wenner.pkl"
FICHIER_TRAIN_TEST = "train_test_split.pkl"
FICHIER_HISTORIQUE = "historique_predictions_wenner.csv"

# ================================
# IMPORTATION DES CLASSES DU MODÈLE
# ================================

try:
    from model_hybride import HybridRegressor, charger_modele, charger_train_test, evaluer_modele
except ImportError:
    print("Impossible d'importer le module model_hybride.py")
    print("Assurez-vous que le fichier model_hybride.py est dans le même dossier")
    # Définition simplifiée pour le cas où le fichier n'existe pas
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.base import BaseEstimator, RegressorMixin
    
    class HybridRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, rf_weight=0.7, knn_weight=0.3, **kwargs):
            self.rf_weight = rf_weight
            self.knn_weight = knn_weight
            self.rf_model = None
            self.knn_model = None
            self.is_fitted = False
            
        def fit(self, X, y):
            self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rf_model.fit(X, y)
            self.knn_model = KNeighborsRegressor(n_neighbors=5)
            self.knn_model.fit(X, y)
            self.is_fitted = True
            return self
            
        def predict(self, X):
            if not self.is_fitted:
                raise ValueError("Le modèle n'a pas été entraîné")
            rf_pred = self.rf_model.predict(X)
            knn_pred = self.knn_model.predict(X)
            return self.rf_weight * rf_pred + self.knn_weight * knn_pred
    
    def charger_modele():
        try:
            return joblib.load(FICHIER_MODELE), joblib.load(FICHIER_SCALER)
        except:
            return None, None
    
    def charger_train_test():
        try:
            data = joblib.load(FICHIER_TRAIN_TEST)
            return data.get('X_train'), data.get('X_test'), data.get('y_train'), data.get('y_test')
        except:
            return None, None, None, None
    
    def evaluer_modele(model, scaler, X, y):
        try:
            Xs = scaler.transform(X)
            y_pred = model.predict(Xs)
            from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            return mae, rmse, r2
        except:
            return None, None, None

# ================================
# FONCTIONS DE CHARGEMENT
# ================================

@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage de l'API"""
    global model, scaler, X_train_global, X_test_global, y_train_global, y_test_global
    
    print("Chargement du modèle de prédiction de résistivité...")
    
    # Charger le modèle
    model, scaler = charger_modele()
    
    if model is None:
        print("⚠️ Aucun modèle trouvé. Veuillez d'abord entraîner le modèle avec model_hybride.py")
        print("   ou placez les fichiers modele_hybride_wenner.pkl et scaler_wenner.pkl dans le dossier")
    
    # Charger les ensembles train/test
    X_train_global, X_test_global, y_train_global, y_test_global = charger_train_test()
    
    print("✅ API prête !")

# ================================
# ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API de Prédiction de Résistivité du Sol",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Prédire la résistivité pour des coordonnées",
            "/predict/csv": "POST - Prédire à partir d'un fichier CSV",
            "/model/info": "GET - Informations sur le modèle",
            "/model/retrain": "POST - Réentraîner le modèle",
            "/health": "GET - Vérifier l'état de l'API"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Obtenir les informations sur le modèle"""
    if model is None:
        raise HTTPException(status_code=404, detail="Modèle non chargé")
    
    return ModelInfo(
        type_modele="Hybrid Regressor (Random Forest + KNN)",
        poids_rf=model.rf_weight if hasattr(model, 'rf_weight') else 0.7,
        poids_knn=model.knn_weight if hasattr(model, 'knn_weight') else 0.3,
        rf_n_estimators=model.rf_model.n_estimators if hasattr(model, 'rf_model') and model.rf_model else 100,
        rf_max_depth=model.rf_model.max_depth if hasattr(model, 'rf_model') and model.rf_model else 10,
        knn_n_neighbors=model.knn_model.n_neighbors if hasattr(model, 'knn_model') and model.knn_model else 5,
        knn_weights=model.knn_model.weights if hasattr(model, 'knn_model') and model.knn_model else 'uniform',
        est_entraine=model.is_fitted if hasattr(model, 'is_fitted') else False,
        date_chargement=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prédire la résistivité pour une liste de points
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # Convertir les points en DataFrame
        points_df = pd.DataFrame([p.dict() for p in request.points])
        
        # Prédire
        X = points_df[["latitude", "longitude", "altitude"]]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Construire les résultats
        results = []
        for i, point in enumerate(request.points):
            results.append(PredictionResult(
                latitude=point.latitude,
                longitude=point.longitude,
                altitude=point.altitude,
                resistivite_wenner=float(predictions[i])
            ))
        
        # Sauvegarder dans l'historique
        points_df["wenner"] = predictions
        if os.path.exists(FICHIER_HISTORIQUE):
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            points_df = pd.concat([historique, points_df], ignore_index=True)
        points_df.to_csv(FICHIER_HISTORIQUE, index=False)
        
        return PredictionResponse(
            predictions=results,
            nombre_points=len(results),
            resistivite_moyenne=float(np.mean(predictions)),
            resistivite_min=float(np.min(predictions)),
            resistivite_max=float(np.max(predictions)),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Prédire la résistivité à partir d'un fichier CSV
    Le fichier doit contenir les colonnes: latitude, longitude, altitude
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    try:
        # Lire le fichier CSV
        content = await file.read()
        points_df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=";")
        
        # Vérifier les colonnes
        colonnes_requises = ["latitude", "longitude", "altitude"]
        for col in colonnes_requises:
            if col not in points_df.columns:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Colonne '{col}' manquante dans le fichier"
                )
        
        # Convertir les colonnes en float
        for col in colonnes_requises:
            points_df[col] = points_df[col].astype(str).str.replace(",", ".").astype(float)
        
        # Prédire
        X = points_df[colonnes_requises]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        points_df["wenner"] = predictions
        
        # Sauvegarder dans l'historique
        if os.path.exists(FICHIER_HISTORIQUE):
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            points_df = pd.concat([historique, points_df], ignore_index=True)
        points_df.to_csv(FICHIER_HISTORIQUE, index=False)
        
        return {
            "message": "Prédictions effectuées avec succès",
            "nombre_points": len(points_df),
            "predictions": points_df.to_dict(orient="records"),
            "resistivite_moyenne": float(predictions.mean()),
            "resistivite_min": float(predictions.min()),
            "resistivite_max": float(predictions.max()),
            "timestamp": datetime.now().isoformat()
        }
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Le fichier est vide")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain", response_model=ReentrainementResponse)
async def retrain_model():
    """
    Réentraîner le modèle avec les données d'entraînement disponibles
    (Nécessite d'avoir le fichier original des données)
    """
    global model, scaler
    
    try:
        # Vérifier si les données d'entraînement existent
        if X_train_global is None:
            raise HTTPException(
                status_code=404, 
                detail="Données d'entraînement non trouvées"
            )
        
        # Créer le jeu de données complet
        data_train = X_train_global.copy()
        data_train["wenner"] = y_train_global
        
        # Charger l'historique
        if os.path.exists(FICHIER_HISTORIQUE):
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            if not historique.empty:
                data_train = pd.concat([data_train, historique], ignore_index=True)
        
        X_complete = data_train[["latitude", "longitude", "altitude"]]
        y_complete = data_train["wenner"]
        
        # Réentraîner
        from sklearn.preprocessing import StandardScaler
        scaler_nouveau = StandardScaler()
        X_scaled = scaler_nouveau.fit_transform(X_complete)
        
        model_nouveau = HybridRegressor(
            rf_weight=0.7,
            knn_weight=0.3,
            rf_n_estimators=100,
            rf_max_depth=10,
            rf_min_samples_split=10,
            rf_min_samples_leaf=4,
            knn_n_neighbors=11,
            knn_weights='uniform'
        )
        model_nouveau.fit(X_scaled, y_complete)
        
        # Évaluer
        mae, rmse, r2 = evaluer_modele(model_nouveau, scaler_nouveau, X_complete, y_complete)
        
        # Sauvegarder
        joblib.dump(model_nouveau, FICHIER_MODELE)
        joblib.dump(scaler_nouveau, FICHIER_SCALER)
        
        # Mettre à jour les variables globales
        model = model_nouveau
        scaler = scaler_nouveau
        
        return ReentrainementResponse(
            message="Modèle réentraîné avec succès",
            nombre_points=len(data_train),
            performance_r2=r2 if r2 is not None else 0.0,
            performance_mae=mae if mae is not None else 0.0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/historique")
async def get_historique(limit: int = 100):
    """
    Obtenir l'historique des prédictions
    """
    try:
        if os.path.exists(FICHIER_HISTORIQUE):
            historique = pd.read_csv(FICHIER_HISTORIQUE)
            if len(historique) > limit:
                historique = historique.tail(limit)
            return {
                "total_points": len(historique),
                "historique": historique.to_dict(orient="records")
            }
        else:
            return {"message": "Aucun historique trouvé", "total_points": 0, "historique": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ================================
# LANCEMENT DE L'API
# ================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )