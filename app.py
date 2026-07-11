"""
API REST pour la prédiction de résistivité du sol
Utilise le modèle hybride Random Forest + KNN - Méthode Wenner
Avec réentraînement automatique - Compatible Render
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import os
import threading
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================================
# CLASSE DU MODELE HYBRIDE
# ================================

class HybridRegressor(BaseEstimator, RegressorMixin):
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
            n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split, min_samples_leaf=self.rf_min_samples_leaf,
            random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X, y)
        self.knn_model = KNeighborsRegressor(n_neighbors=self.knn_n_neighbors, weights=self.knn_weights, n_jobs=-1)
        self.knn_model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Le modèle n'a pas encore été entraîné")
        return self.rf_weight * self.rf_model.predict(X) + self.knn_weight * self.knn_model.predict(X)

# ================================
# INITIALISATION
# ================================

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_files/modele_hybride_wenner.pkl'
SCALER_PATH = 'model_files/scaler_wenner.pkl'
HISTORY_PATH = 'model_files/historique_predictions.csv'

model = None
scaler = None
nouveau_points_count = 0
REENTRAINEMENT_SEUIL = 20
is_retraining = False

def load_model():
    global model, scaler
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info("✅ Modèle et scaler chargés avec succès")
            return True
        else:
            logger.warning(f"⚠️ Fichiers modèles non trouvés dans model_files/")
            return False
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return False

def reentrainer_modele():
    global model, scaler, is_retraining
    is_retraining = True
    
    try:
        logger.info("🔄 Début du réentraînement automatique...")
        
        if os.path.exists(HISTORY_PATH):
            data = pd.read_csv(HISTORY_PATH)
        else:
            logger.error("❌ Aucune donnée disponible pour le réentraînement")
            is_retraining = False
            return
        
        if len(data) < 10:
            logger.warning("⚠️ Pas assez de données pour réentraîner (< 10 points)")
            is_retraining = False
            return
        
        X = data[["latitude", "longitude", "altitude"]]
        y = data["wenner"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = HybridRegressor(
            rf_weight=0.7, knn_weight=0.3,
            rf_n_estimators=100, rf_max_depth=10,
            rf_min_samples_split=10, rf_min_samples_leaf=4,
            knn_n_neighbors=11, knn_weights='uniform'
        )
        model.fit(X_scaled, y)
        
        os.makedirs('model_files', exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        
        logger.info(f"✅ Modèle réentraîné avec {len(data)} points et sauvegardé !")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du réentraînement: {e}")
    finally:
        is_retraining = False

def save_to_history(prediction):
    try:
        df = pd.DataFrame([prediction])
        os.makedirs('model_files', exist_ok=True)
        if os.path.exists(HISTORY_PATH):
            df.to_csv(HISTORY_PATH, mode='a', header=False, index=False)
        else:
            df.to_csv(HISTORY_PATH, index=False)
    except Exception as e:
        logger.error(f"Erreur sauvegarde historique: {e}")

def validate_coordinates(lat, lon, alt):
    try:
        lat, lon, alt = float(lat), float(lon), float(alt)
        if not (-90 <= lat <= 90): return False, f"Latitude invalide: {lat}"
        if not (-180 <= lon <= 180): return False, f"Longitude invalide: {lon}"
        if alt < -500 or alt > 9000: return False, f"Altitude invalide: {alt}"
        return True, None
    except ValueError as e:
        return False, str(e)

# ================================
# ROUTES API
# ================================

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'name': 'API de prédiction de résistivité du sol',
        'version': '2.0.0',
        'endpoints': {
            '/health': 'Vérification',
            '/predict': 'Prédiction (POST)',
            '/history': 'Historique',
            '/model/info': 'Infos modèle',
            '/model/status': 'Statut réentraînement',
            '/model/retrain': 'Forcer réentraînement (POST)'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'is_retraining': is_retraining,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    global nouveau_points_count
    
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Données JSON requises'}), 400
        
        if 'latitude' in data and 'longitude' in data and 'altitude' in data:
            lat, lon, alt = data['latitude'], data['longitude'], data['altitude']
            is_valid, error_msg = validate_coordinates(lat, lon, alt)
            if not is_valid:
                return jsonify({'success': False, 'error': error_msg}), 400
            
            X = np.array([[float(lat), float(lon), float(alt)]])
            X_scaled = scaler.transform(X)
            prediction = float(model.predict(X_scaled)[0])
            
            result = {
                'latitude': float(lat), 'longitude': float(lon), 'altitude': float(alt),
                'resistivity': round(prediction, 2), 'unit': 'Ohm.m',
                'timestamp': datetime.now().isoformat()
            }
            save_to_history(result)
            
            nouveau_points_count += 1
            if nouveau_points_count >= REENTRAINEMENT_SEUIL and not is_retraining:
                logger.info(f"🔔 Seuil atteint ({nouveau_points_count}/{REENTRAINEMENT_SEUIL})")
                thread = threading.Thread(target=reentrainer_modele)
                thread.start()
                nouveau_points_count = 0
            
            return jsonify({'success': True, 'prediction': result})
        
        elif 'points' in data:
            results = []
            for point in data['points']:
                lat, lon, alt = point.get('latitude'), point.get('longitude'), point.get('altitude')
                if lat is None or lon is None or alt is None: continue
                is_valid, _ = validate_coordinates(lat, lon, alt)
                if not is_valid: continue
                
                X = np.array([[float(lat), float(lon), float(alt)]])
                X_scaled = scaler.transform(X)
                prediction = float(model.predict(X_scaled)[0])
                
                result = {
                    'latitude': float(lat), 'longitude': float(lon), 'altitude': float(alt),
                    'resistivity': round(prediction, 2), 'unit': 'Ohm.m',
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                save_to_history(result)
                nouveau_points_count += 1
            
            if nouveau_points_count >= REENTRAINEMENT_SEUIL and not is_retraining:
                thread = threading.Thread(target=reentrainer_modele)
                thread.start()
                nouveau_points_count = 0
            
            return jsonify({'success': True, 'predictions': results, 'count': len(results)})
        
        else:
            return jsonify({'success': False, 'error': 'Format invalide'}), 400
            
    except Exception as e:
        logger.error(f"Erreur prédiction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        if os.path.exists(HISTORY_PATH):
            df = pd.read_csv(HISTORY_PATH)
            return jsonify({'success': True, 'history': df.to_dict('records'), 'count': len(df)})
        return jsonify({'success': True, 'history': [], 'count': 0})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'success': False, 'error': 'Modèle non chargé'}), 500
    try:
        info = {
            'type': 'Hybrid Random Forest + KNN',
            'method': 'Wenner',
            'features': ['latitude', 'longitude', 'altitude'],
            'unit': 'Ohm.m'
        }
        if hasattr(model, 'rf_model') and model.rf_model is not None:
            info['rf_n_estimators'] = model.rf_model.n_estimators
        if hasattr(model, 'knn_model') and model.knn_model is not None:
            info['knn_n_neighbors'] = model.knn_model.n_neighbors
        return jsonify({'success': True, 'info': info})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    return jsonify({
        'success': True,
        'model_loaded': model is not None,
        'is_retraining': is_retraining,
        'nouveaux_points': nouveau_points_count,
        'seuil_reentrainement': REENTRAINEMENT_SEUIL,
        'progression': f'{nouveau_points_count}/{REENTRAINEMENT_SEUIL}'
    })

@app.route('/model/retrain', methods=['POST'])
def force_retrain():
    if is_retraining:
        return jsonify({'success': False, 'message': 'Réentraînement déjà en cours'})
    
    thread = threading.Thread(target=reentrainer_modele)
    thread.start()
    return jsonify({'success': True, 'message': 'Réentraînement lancé'})

# ================================
# DÉMARRAGE
# ================================

if __name__ == '__main__':
    os.makedirs('model_files', exist_ok=True)
    model_loaded = load_model()
    
    print("\n" + "="*55)
    print("  🌍 API DE PRÉDICTION DE RÉSISTIVITÉ DU SOL")
    print("     Méthode Wenner - Réentraînement Auto")
    print("="*55)
    print(f"  Modèle chargé : {'✅ OUI' if model_loaded else '❌ NON'}")
    print(f"  Seuil réentraînement : {REENTRAINEMENT_SEUIL} points")
    print("="*55 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)