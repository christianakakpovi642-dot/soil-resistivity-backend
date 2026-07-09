"""
Script de test pour l'API de prédiction de résistivité
"""

import requests
import json

# URL de base de l'API
BASE_URL = "http://localhost:8000"

def test_health():
    """Tester le endpoint /health"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health check: {response.json()}")
    return response.json()

def test_predict():
    """Tester le endpoint /predict"""
    data = {
        "points": [
            {"latitude": 48.8584, "longitude": 2.2945, "altitude": 324},
            {"latitude": 48.8600, "longitude": 2.3000, "altitude": 320},
            {"latitude": 48.8550, "longitude": 2.2900, "altitude": 330}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json=data
    )
    
    print("\nRésultats des prédictions:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

def test_model_info():
    """Tester le endpoint /model/info"""
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"\nInformations du modèle:")
    print(json.dumps(response.json(), indent=2))
    return response.json()

if __name__ == "__main__":
    print("Test de l'API de prédiction de résistivité")
    print("="*50)
    
    # Tester la santé de l'API
    test_health()
    
    # Tester les prédictions
    test_predict()
    
    # Tester les informations du modèle
    test_model_info()