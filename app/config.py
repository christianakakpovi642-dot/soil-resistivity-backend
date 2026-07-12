import os
from dotenv import load_dotenv

# Charge automatiquement les variables depuis backend/.env si le fichier existe.
# N'écrase jamais une variable déjà définie dans l'environnement système
# (utile en production, où les vraies variables d'env priment).
load_dotenv()

# Modèle
MODEL_PATH = os.getenv("MODEL_PATH", "models/modele_hybride_wenner.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler_wenner.pkl")

# Base de données (historique + données de seed)
DB_PATH = os.getenv("DB_PATH", "models/history.db")

# Fichier CSV optionnel pour amorcer la base avec tes données d'entraînement
# initiales (mêmes colonnes que ton script : latitude;longitude;altitude;wenner)
INITIAL_DATA_CSV = os.getenv("INITIAL_DATA_CSV", "models/initial_training_data.csv")

# Seuils de ré-entraînement (mêmes valeurs par défaut que ton script original)
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", 20))
PERFORMANCE_R2_THRESHOLD = float(os.getenv("PERFORMANCE_R2_THRESHOLD", 0.60))

# Poids du modèle hybride (mêmes valeurs par défaut que ton script original)
RF_WEIGHT = float(os.getenv("RF_WEIGHT", 0.7))
KNN_WEIGHT = float(os.getenv("KNN_WEIGHT", 0.3))

# Clé optionnelle pour protéger les endpoints d'administration
# (laisse vide en dev, définis-la en production)
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

# ================================
# Alertes de maintenance (email)
# ================================
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO", "")
# Sévérité minimale déclenchant un envoi d'email : "info" | "warning" | "critical"
ALERT_EMAIL_MIN_SEVERITY = os.getenv("ALERT_EMAIL_MIN_SEVERITY", "critical")

# Nombre de jours sans ré-entraînement (avec données en attente) avant alerte
MODEL_STALE_DAYS = int(os.getenv("MODEL_STALE_DAYS", 30))
# Nombre d'erreurs consécutives sur /predict avant alerte de santé serveur
MAX_CONSECUTIVE_PREDICT_ERRORS = int(os.getenv("MAX_CONSECUTIVE_PREDICT_ERRORS", 3))

# ================================
# Version de l'application mobile
# ================================
# Valeurs de départ (modifiables ensuite via /admin/app-config sans redéployer)
DEFAULT_LATEST_APP_VERSION = os.getenv("DEFAULT_LATEST_APP_VERSION", "1.0.0")
DEFAULT_MIN_SUPPORTED_APP_VERSION = os.getenv("DEFAULT_MIN_SUPPORTED_APP_VERSION", "1.0.0")
DEFAULT_UPDATE_URL = os.getenv("DEFAULT_UPDATE_URL", "")
DEFAULT_UPDATE_MESSAGE = os.getenv(
    "DEFAULT_UPDATE_MESSAGE", "Une nouvelle version de l'application est disponible."
)
