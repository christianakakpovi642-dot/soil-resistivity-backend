# Soil Resistivity API

Backend FastAPI qui expose ton modèle hybride (Random Forest + KNN) de prédiction
de résistivité du sol (méthode Wenner) sous forme d'API REST.

## 1. Structure du projet

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py            # Endpoints FastAPI
│   ├── model_service.py   # Chargement du modèle + logique de prédiction
│   ├── hybrid_model.py    # Définition de la classe HybridRegressor
│   └── schemas.py         # Schémas de validation (Pydantic)
├── models/
│   ├── modele_hybride_wenner.pkl   ← À COPIER ICI
│   └── scaler_wenner.pkl           ← À COPIER ICI
├── requirements.txt
└── README.md
```

## 2. Installation

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Copier ton modèle entraîné

Ton script génère déjà `modele_hybride_wenner.pkl` et `scaler_wenner.pkl`.
Copie-les tels quels dans `backend/models/` :

```bash
cp modele_hybride_wenner.pkl backend/models/
cp scaler_wenner.pkl backend/models/
```

**Important** : la classe `HybridRegressor` dans `app/hybrid_model.py` doit
avoir exactement la même structure que celle utilisée lors de l'entraînement
(mêmes attributs, même logique). Je l'ai recopiée à l'identique depuis ton
script — si tu modifies la classe côté entraînement, répercute le changement ici.

## 4. Lancer le serveur en local

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` est indispensable pour que ton téléphone (sur le même
  Wi-Fi) ou l'émulateur Android puisse atteindre le serveur.
- Documentation interactive auto-générée : http://localhost:8000/docs

## 5. Tester rapidement

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"latitude": 6.1319, "longitude": 1.2228, "altitude": 50.0}'
```

Réponse attendue :
```json
{
  "latitude": 6.1319,
  "longitude": 1.2228,
  "altitude": 50.0,
  "resistivity": 123.45,
  "unit": "Ohm.m"
}
```

## 6. Endpoints disponibles

| Méthode | Route            | Description                              |
|---------|-------------------|-------------------------------------------|
| GET     | `/health`         | Vérifie que l'API et le modèle sont prêts |
| POST    | `/predict`        | Prédiction pour un point unique           |
| POST    | `/predict/batch`  | Prédiction pour plusieurs points          |

### Exemple `/predict/batch`
```json
{
  "points": [
    {"latitude": 6.1319, "longitude": 1.2228, "altitude": 50.0},
    {"latitude": 6.14, "longitude": 1.23, "altitude": 55.0}
  ]
}
```

## 7. Connecter ton app Flutter

Reprends le `ApiService` déjà mis en place côté Flutter et pointe `baseUrl` vers :

- **Émulateur Android** → `http://10.0.2.2:8000`
- **Téléphone physique (même Wi-Fi que ton PC)** → `http://<IP_LOCALE_DE_TON_PC>:8000`
  (trouve ton IP avec `ipconfig` sous Windows, cherche `IPv4`)
- **Backend déployé en ligne** → `https://ton-domaine.com`

N'oublie pas la configuration `network_security_config.xml` si tu restes en
HTTP pendant le développement (voir échange précédent).

## 8. Déploiement en production

Options simples et gratuites/peu coûteuses pour héberger ce backend :

- **Render** (render.com) : déploiement direct depuis un repo GitHub, gratuit pour démarrer
- **Railway** (railway.app) : très simple, bon plan gratuit
- **Fly.io** : bon pour les apps avec besoin de perf

Commande de démarrage à utiliser sur ces plateformes :
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Pense à uploader `models/modele_hybride_wenner.pkl` et `models/scaler_wenner.pkl`
avec ton dépôt (ou à les stocker dans un stockage externe si les fichiers sont volumineux).

## 9. Ré-entraînement automatique

Le backend ré-entraîne maintenant le modèle **tout seul**, déclenché par les
prédictions envoyées depuis l'app. Aucun redémarrage du serveur n'est nécessaire :
le nouveau modèle est publié en mémoire immédiatement après le ré-entraînement.

### Comment ça marche

1. Chaque appel à `/predict` (ou `/predict/batch`) enregistre le point dans une
   base SQLite (`models/history.db`) :
   - si l'app envoie `measured_resistivity` (vraie mesure Wenner de terrain),
     cette valeur est stockée comme donnée d'entraînement **fiable**
   - sinon, la prédiction du modèle elle-même est stockée comme donnée
     d'apprentissage (comportement identique à ton script original)
2. Juste après avoir répondu au client, une vérification est lancée **en tâche
   de fond** (elle ne ralentit jamais la réponse) :
   - déclenchement si le nombre de nouveaux points ≥ `RETRAIN_THRESHOLD` (20 par défaut)
   - OU si la performance R² du modèle actuel tombe sous `PERFORMANCE_R2_THRESHOLD` (0.60 par défaut)
3. Si déclenché : ré-entraînement sur (données de seed + historique complet),
   sauvegarde sur disque, et publication immédiate en mémoire

### Amorcer avec tes données d'entraînement initiales

Pour que le modèle ne parte pas de zéro, copie ton fichier `resistivite_wenner.csv`
original dans `backend/models/initial_training_data.csv` (mêmes colonnes :
`latitude;longitude;altitude;wenner`). Il sera importé automatiquement dans la
base au premier démarrage du serveur.

### Variables d'environnement disponibles

| Variable | Défaut | Description |
|---|---|---|
| `RETRAIN_THRESHOLD` | `20` | Nombre de nouveaux points avant ré-entraînement forcé |
| `PERFORMANCE_R2_THRESHOLD` | `0.60` | Seuil de R² en dessous duquel on ré-entraîne |
| `RF_WEIGHT` / `KNN_WEIGHT` | `0.7` / `0.3` | Poids du modèle hybride |
| `ADMIN_API_KEY` | *(vide)* | Si définie, protège `/admin/*` par le header `X-Admin-Key` |
| `INITIAL_DATA_CSV` | `models/initial_training_data.csv` | Fichier de seed optionnel |

### Endpoints d'administration

| Méthode | Route | Description |
|---|---|---|
| GET | `/admin/stats` | Nombre de points collectés, dernier ré-entraînement, etc. |
| POST | `/admin/retrain` | Force un ré-entraînement immédiat (bloquant) |

```bash
curl http://localhost:8000/admin/stats
curl -X POST http://localhost:8000/admin/retrain
```

Si `ADMIN_API_KEY` est définie, ajoute l'en-tête à chaque appel :
```bash
curl -X POST http://localhost:8000/admin/retrain -H "X-Admin-Key: ta-cle-secrete"
```

### Points d'attention

- **Un seul worker Uvicorn.** Le modèle vit en mémoire dans le process Python.
  Si tu déploies avec plusieurs workers (`--workers 4`), chaque worker aura sa
  propre copie du modèle et ne saura pas que les autres ont ré-entraîné. Pour
  ce cas d'usage (petite app, faible trafic), reste sur `--workers 1`. Si tu as
  besoin de scaler plus tard, il faudra migrer vers un stockage de modèle
  partagé (ex: S3 + rechargement périodique).
- **Historique jamais perdu.** Contrairement au script original qui vidait le
  CSV d'historique après chaque ré-entraînement (et perdait ainsi ces points
  pour les cycles suivants), ce backend recombine systématiquement *tout*
  l'historique + les données de seed à chaque ré-entraînement.
- **Auto-apprentissage sur prédictions non confirmées.** Si l'app n'envoie
  jamais `measured_resistivity`, le modèle s'entraîne sur ses propres
  prédictions — un risque classique de "chambre d'écho" qui peut renforcer des
  biais existants plutôt que les corriger. Encourage l'envoi de vraies mesures
  de terrain chaque fois que possible pour un ré-entraînement réellement utile.
- **Pas de nouvelle recherche d'hyperparamètres à chaque cycle.** Le
  `GridSearchCV` de ton script original est trop coûteux pour tourner à chaque
  ré-entraînement automatique ; les meilleurs hyperparamètres trouvés
  précédemment sont réutilisés tels quels (ajustés dynamiquement si le jeu de
  données est encore petit).
