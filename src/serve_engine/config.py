from __future__ import annotations

import os
from pathlib import Path

SERVE_DIR = Path(os.environ.get("SERVE_HOME", Path.home() / ".serve"))
MODELS_DIR = SERVE_DIR / "models"
LOGS_DIR = SERVE_DIR / "logs"
DB_PATH = SERVE_DIR / "db.sqlite"
SOCK_PATH = SERVE_DIR / "sock"

DEFAULT_PUBLIC_HOST = "127.0.0.1"
DEFAULT_PUBLIC_PORT = 11500

DOCKER_NETWORK_NAME = "serve-engines"
