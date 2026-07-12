"""
Envoi d'email pour les alertes critiques, via SMTP standard (Gmail, Outlook,
SendGrid SMTP, etc. — n'importe quel fournisseur SMTP classique convient).

Si aucune configuration SMTP n'est fournie (variables d'environnement vides),
l'envoi est simplement ignoré (log + return False) : l'alerte reste visible
via /admin/alerts même sans email configuré.
"""

import logging
import smtplib
from email.mime.text import MIMEText

from app.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_USE_TLS,
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_TO,
)

logger = logging.getLogger("email_service")


def is_configured() -> bool:
    return bool(SMTP_HOST and ALERT_EMAIL_TO)


def send_email(subject: str, body: str) -> bool:
    if not is_configured():
        logger.info(
            "Email non envoyé (SMTP_HOST ou ALERT_EMAIL_TO manquant) — "
            "alerte disponible uniquement via /admin/alerts. Sujet: %s",
            subject,
        )
        return False

    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = ALERT_EMAIL_FROM or SMTP_USER
        msg["To"] = ALERT_EMAIL_TO

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            if SMTP_USE_TLS:
                server.starttls()
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(msg["From"], [ALERT_EMAIL_TO], msg.as_string())

        logger.info("Email d'alerte envoyé : %s", subject)
        return True
    except Exception as e:
        logger.error("Échec de l'envoi de l'email d'alerte : %s", e)
        return False
