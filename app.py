from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

class HybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, rf_weight=0.6, knn_weight=0.4, rf_n_estimators=200, rf_max_depth=20,
                 rf_min_samples_split=5, rf_min_samples_leaf=2, knn_n_neighbors=5, knn_weights='distance'):
        self.rf_weight = rf_weight; self.knn_weight = knn_weight
        self.rf_n_estimators = rf_n_estimators; self.rf_max_depth = rf_max_depth
        self.rf_min_samples_split = rf_min_samples_split; self.rf_min_samples_leaf = rf_min_samples_leaf
        self.knn_n_neighbors = knn_n_neighbors; self.knn_weights = knn_weights
        self.rf_model = None; self.knn_model = None; self.is_fitted = False
        
    def fit(self, X, y):
        self.rf_model = RandomForestRegressor(n_estimators=self.rf_n_estimators, max_depth=self.rf_max_depth,
            min_samples_split=self.rf_min_samples_split, min_samples_leaf=self.rf_min_samples_leaf, random_state=42, n_jobs=1)
        self.rf_model.fit(X, y)
        self.knn_model = KNeighborsRegressor(n_neighbors=self.knn_n_neighbors, weights=self.knn_weights, n_jobs=1)
        self.knn_model.fit(X, y)
        self.is_fitted = True; return self
    
    def predict(self, X):
        return self.rf_weight * self.rf_model.predict(X) + self.knn_weight * self.knn_model.predict(X)

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'model_files/modele_hybride_wenner.pkl'
SCALER_PATH = 'model_files/scaler_wenner.pkl'
model = None; scaler = None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH); scaler = joblib.load(SCALER_PATH)
        return True
    return False

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'success': False, 'error': 'Modele non charge'}), 500
    try:
        data = request.get_json()
        lat = float(data.get('latitude', 0)); lon = float(data.get('longitude', 0)); alt = float(data.get('altitude', 0))
        X = np.array([[lat, lon, alt]]); X_scaled = scaler.transform(X)
        r = float(model.predict(X_scaled)[0])
        return jsonify({'success': True, 'prediction': {'latitude': lat, 'longitude': lon, 'altitude': alt, 'resistivity': round(r, 2), 'unit': 'Ohm.m'}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs('model_files', exist_ok=True)
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)