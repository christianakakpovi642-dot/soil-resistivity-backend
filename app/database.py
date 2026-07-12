"""
Persistance de l'historique des prédictions et des données d'entraînement.

Utilise SQLite plutôt que des fichiers CSV (comme dans le script original) car
le backend reçoit des requêtes concurrentes — SQLite + verrou applicatif évite
les écritures corrompues que des CSV partagés provoqueraient sous charge.
"""

import os
import threading
import sqlite3
import pandas as pd

from app.config import (
    DB_PATH,
    INITIAL_DATA_CSV,
    DEFAULT_LATEST_APP_VERSION,
    DEFAULT_MIN_SUPPORTED_APP_VERSION,
    DEFAULT_UPDATE_URL,
    DEFAULT_UPDATE_MESSAGE,
)

_lock = threading.Lock()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crée les tables si nécessaire, et amorce les données de seed depuis un CSV."""
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            altitude REAL NOT NULL,
            wenner REAL NOT NULL,
            source TEXT NOT NULL DEFAULT 'predicted',   -- 'predicted' ou 'measured'
            used_in_training INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_seed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            altitude REAL NOT NULL,
            wenner REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL DEFAULT (datetime('now')),
            n_points_total INTEGER,
            n_new_points_after INTEGER,
            r2_train REAL,
            mae_train REAL,
            rmse_train REAL,
            duration_s REAL,
            reason TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            severity TEXT NOT NULL,               -- 'info' | 'warning' | 'critical'
            message TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()

    # Seeding des valeurs de version par défaut si la table est vide
    existing_keys = {row["key"] for row in conn.execute("SELECT key FROM app_config")}
    defaults = {
        "latest_version": DEFAULT_LATEST_APP_VERSION,
        "min_supported_version": DEFAULT_MIN_SUPPORTED_APP_VERSION,
        "update_url": DEFAULT_UPDATE_URL,
        "update_message": DEFAULT_UPDATE_MESSAGE,
    }
    for key, value in defaults.items():
        if key not in existing_keys:
            conn.execute("INSERT INTO app_config (key, value) VALUES (?, ?)", (key, value))
    conn.commit()

    # Amorçage unique : si la table de seed est vide et qu'un CSV existe, on l'importe.
    count = conn.execute("SELECT COUNT(*) AS c FROM training_seed").fetchone()["c"]
    if count == 0 and os.path.exists(INITIAL_DATA_CSV):
        try:
            df = pd.read_csv(INITIAL_DATA_CSV, sep=None, engine="python")
            df.columns = [c.strip().lower() for c in df.columns]
            required = {"latitude", "longitude", "altitude", "wenner"}
            if required.issubset(set(df.columns)):
                for col in required:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
                rows = list(df[["latitude", "longitude", "altitude", "wenner"]].itertuples(index=False, name=None))
                conn.executemany(
                    "INSERT INTO training_seed (latitude, longitude, altitude, wenner) VALUES (?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
        except Exception:
            pass  # amorçage best-effort : on ne bloque pas le démarrage du serveur

    conn.close()


def add_prediction(latitude: float, longitude: float, altitude: float, wenner: float, source: str = "predicted"):
    with _lock:
        conn = get_connection()
        conn.execute(
            "INSERT INTO predictions_history (latitude, longitude, altitude, wenner, source) VALUES (?, ?, ?, ?, ?)",
            (latitude, longitude, altitude, wenner, source),
        )
        conn.commit()
        conn.close()


def count_unused_history() -> int:
    conn = get_connection()
    c = conn.execute("SELECT COUNT(*) AS c FROM predictions_history WHERE used_in_training = 0").fetchone()["c"]
    conn.close()
    return c


def get_training_dataframe() -> pd.DataFrame:
    """Combine les données de seed + tout l'historique (mesuré + prédit) pour le ré-entraînement."""
    conn = get_connection()
    seed_df = pd.read_sql_query("SELECT latitude, longitude, altitude, wenner FROM training_seed", conn)
    hist_df = pd.read_sql_query("SELECT latitude, longitude, altitude, wenner FROM predictions_history", conn)
    conn.close()

    frames = [df for df in (seed_df, hist_df) if not df.empty]
    if not frames:
        return pd.DataFrame(columns=["latitude", "longitude", "altitude", "wenner"])
    return pd.concat(frames, ignore_index=True).astype(float)


def mark_history_as_used():
    with _lock:
        conn = get_connection()
        conn.execute("UPDATE predictions_history SET used_in_training = 1 WHERE used_in_training = 0")
        conn.commit()
        conn.close()


def log_retrain(n_points_total, n_new_points_after, r2_train, mae_train, rmse_train, duration_s, reason):
    with _lock:
        conn = get_connection()
        conn.execute(
            """INSERT INTO retrain_log
               (n_points_total, n_new_points_after, r2_train, mae_train, rmse_train, duration_s, reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (n_points_total, n_new_points_after, r2_train, mae_train, rmse_train, duration_s, reason),
        )
        conn.commit()
        conn.close()


def get_stats() -> dict:
    conn = get_connection()
    seed_count = conn.execute("SELECT COUNT(*) AS c FROM training_seed").fetchone()["c"]
    hist_total = conn.execute("SELECT COUNT(*) AS c FROM predictions_history").fetchone()["c"]
    hist_measured = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions_history WHERE source = 'measured'"
    ).fetchone()["c"]
    hist_unused = conn.execute(
        "SELECT COUNT(*) AS c FROM predictions_history WHERE used_in_training = 0"
    ).fetchone()["c"]
    last_retrain_row = conn.execute("SELECT * FROM retrain_log ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    return {
        "seed_points": seed_count,
        "history_total": hist_total,
        "history_measured": hist_measured,
        "history_predicted": hist_total - hist_measured,
        "history_pending_retrain": hist_unused,
        "last_retrain": dict(last_retrain_row) if last_retrain_row else None,
    }


# ================================
# Alertes de maintenance
# ================================

def create_alert(alert_type: str, severity: str, message: str) -> dict | None:
    """
    Crée une alerte, sauf si une alerte non résolue du même type existe déjà
    (évite le spam de doublons pour un problème déjà signalé et pas encore réglé).
    Retourne l'alerte créée, ou None si elle a été dédupliquée.
    """
    with _lock:
        conn = get_connection()
        existing = conn.execute(
            "SELECT id FROM alerts WHERE type = ? AND resolved = 0", (alert_type,)
        ).fetchone()
        if existing:
            conn.close()
            return None

        cur = conn.execute(
            "INSERT INTO alerts (type, severity, message) VALUES (?, ?, ?)",
            (alert_type, severity, message),
        )
        conn.commit()
        alert_id = cur.lastrowid
        row = conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        conn.close()
        return dict(row)


def resolve_alerts_of_type(alert_type: str):
    """Marque comme résolues toutes les alertes non résolues d'un type donné."""
    with _lock:
        conn = get_connection()
        conn.execute(
            "UPDATE alerts SET resolved = 1, resolved_at = datetime('now') "
            "WHERE type = ? AND resolved = 0",
            (alert_type,),
        )
        conn.commit()
        conn.close()


def resolve_alert(alert_id: int) -> bool:
    with _lock:
        conn = get_connection()
        cur = conn.execute(
            "UPDATE alerts SET resolved = 1, resolved_at = datetime('now') "
            "WHERE id = ? AND resolved = 0",
            (alert_id,),
        )
        conn.commit()
        updated = cur.rowcount > 0
        conn.close()
        return updated


def get_alerts(resolved: bool | None = None, severity: str | None = None) -> list[dict]:
    conn = get_connection()
    query = "SELECT * FROM alerts WHERE 1=1"
    params: list = []
    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)
    if severity is not None:
        query += " AND severity = ?"
        params.append(severity)
    query += " ORDER BY id DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ================================
# Configuration de version de l'app
# ================================

def get_app_config() -> dict:
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM app_config").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def set_app_config(updates: dict):
    with _lock:
        conn = get_connection()
        for key, value in updates.items():
            if value is None:
                continue
            conn.execute(
                "INSERT INTO app_config (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
        conn.commit()
        conn.close()
