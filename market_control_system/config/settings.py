"""
settings.py
============

Laedt Secrets (Alpaca API-Keys etc.) ausschliesslich aus Umgebungsvariablen
bzw. einer lokalen ".env"-Datei -- niemals hartkodiert im Code.

".env" liegt im Projekt-Root (market_control_system/.env, neben dieser
config/-Mappe), ist in .gitignore eingetragen und wird nie eingecheckt.
".env.example" im selben Verzeichnis zeigt das erwartete Format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


@dataclass
class AlpacaConfig:
    api_key: str
    secret_key: str
    base_url: str


def load_alpaca_config() -> AlpacaConfig:
    """
    Liest ALPACA_API_KEY / ALPACA_SECRET_KEY / ALPACA_BASE_URL aus der
    Umgebung (bzw. aus .env). Wirft eine klare Fehlermeldung, falls Werte
    fehlen -- fail-fast statt mit leeren Credentials weiterzulaufen und
    erst bei der ersten API-Anfrage einen kryptischen 401 zu bekommen.
    """
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")

    missing = [
        name for name, value in [("ALPACA_API_KEY", api_key), ("ALPACA_SECRET_KEY", secret_key)]
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Fehlende Alpaca-Credentials: {', '.join(missing)}. "
            f"Bitte {_ENV_PATH} anlegen (siehe .env.example im selben Verzeichnis) "
            "und dort ALPACA_API_KEY / ALPACA_SECRET_KEY eintragen."
        )

    return AlpacaConfig(api_key=api_key, secret_key=secret_key, base_url=base_url)


# ---------------------------------------------------------------------------
# Sanity-Check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        cfg = load_alpaca_config()
        print(f"Alpaca-Config geladen: base_url={cfg.base_url}, "
              f"api_key={cfg.api_key[:4]}...{cfg.api_key[-2:]} (maskiert)")
    except RuntimeError as e:
        print(f"Config unvollstaendig: {e}")
