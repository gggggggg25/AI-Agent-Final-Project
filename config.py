"""Shared settings."""
import os
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
OUT = Path("outputs")
TARGET = "G3"
# Google's list showed gemini-3.6-flash as the newest Flash model; override via env var.
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
