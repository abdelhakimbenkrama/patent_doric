"""Paths and .env access shared by every module."""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
STATIC_DIR = ROOT / "static"
# Vercel's filesystem is read-only except /tmp (per instance, not persistent). CACHE_DIR overrides.
CACHE_DIR = Path(os.getenv("CACHE_DIR") or ("/tmp/patent-cache" if os.getenv("VERCEL") else ROOT / "cache"))

PLACEHOLDER = "your_key_here"


def get_env(name: str, default: str = "") -> str:
    # Re-read .env on every call so keys and settings can change without restarting.
    load_dotenv(ENV_FILE, override=True)
    return os.getenv(name, default).strip()


def require_key(name: str) -> str:
    value = get_env(name)
    if not value or value == PLACEHOLDER:
        raise HTTPException(status_code=500, detail=f"{name} is not set. Add it to {ENV_FILE}.")
    return value
