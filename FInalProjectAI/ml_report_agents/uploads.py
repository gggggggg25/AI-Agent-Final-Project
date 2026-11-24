"""Makes files attached in the ADK web chat usable by the tools.

A `before_model_callback` on the coordinator looks for attached data files (CSV/TSV/TXT/XLSX/XLS),
saves them to DATA_DIR/uploads/ and swaps each attachment, in the copy of the conversation sent to the
model, for a short text note with the file name, size and columns. The tools then load it by that name.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from google.genai import types

from . import tools

log = logging.getLogger(__name__)
MAX_BYTES = 25 * 1024 * 1024
UPLOAD_DIR = tools.DATA_DIR / "uploads"


def _detect_ext(data: bytes, mime: str, name: str) -> Optional[str]:
    """Return a file extension for supported tabular files, else None (leave other attachments alone)."""
    if data[:4] == b"\xd0\xcf\x11\xe0":
        return ".xls"
    if data[:2] == b"PK":  # zip container -> xlsx (the only zip-based format we accept)
        return ".xlsx" if (mime.endswith("sheet") or name.lower().endswith(".xlsx") or "excel" in mime or not mime) else None
    low = name.lower()
    for ext in (".csv", ".tsv", ".txt"):
        if low.endswith(ext):
            return ext
    if mime in ("text/csv", "application/csv", "text/plain", "text/tab-separated-values", "application/vnd.ms-excel"):
        return ".tsv" if "tab" in mime else ".csv"
    return None


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "upload"


def _describe(path: Path) -> str:
    try:
        df = tools._read_table(path)
        cols = ", ".join(map(str, df.columns[:60])) + (" ..." if df.shape[1] > 60 else "")
        return f"{len(df)} rows x {df.shape[1]} columns. Columns: {cols}"
    except Exception as e:  # noqa: BLE001
        return f"(could not preview: {e})"


def save_attachments(contents: list) -> list:
    """Returns a copy of `contents` where supported data attachments are saved and replaced by text notes."""
    out = []
    for content in contents or []:
        parts = list(getattr(content, "parts", None) or [])
        changed, new_parts = False, []
        for part in parts:
            blob = getattr(part, "inline_data", None)
            data = getattr(blob, "data", None) if blob else None
            if not data or len(data) > MAX_BYTES:
                new_parts.append(part)
                continue
            mime = (getattr(blob, "mime_type", "") or "").lower()
            display = getattr(blob, "display_name", None) or ""
            ext = _detect_ext(data, mime, display)
            if ext is None:
                new_parts.append(part)
                continue
            stem = Path(display).stem if display else "upload"
            fname = f"{_safe_name(stem)}_{hashlib.sha1(data).hexdigest()[:6]}{ext}"
            path = UPLOAD_DIR / fname
            if not path.exists():
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                log.info("Saved chat attachment to %s", path)
            note = (f"[Attached data file saved as '{fname}' - use source='{fname}'. {_describe(path)}]")
            new_parts.append(types.Part(text=note))
            changed = True
        out.append(types.Content(role=content.role, parts=new_parts) if changed else content)
    return out


def attachments_callback(callback_context, llm_request):
    """ADK before_model_callback. Returns None so the model call proceeds with the rewritten request."""
    try:
        llm_request.contents = save_attachments(llm_request.contents)
    except Exception:  # noqa: BLE001  never block the chat because of an upload problem
        log.exception("Attachment handling failed")
    return None
