"""Download and load the UCI Student Performance data straight from the URL."""
import io
import urllib.request
import zipfile

import pandas as pd

from config import DATA_URL


def _find_and_read(zf, filename):
    """Find a CSV in this zip, recursing into nested zips."""
    for name in zf.namelist():
        if name.endswith(filename):
            with zf.open(name) as f:
                return pd.read_csv(f, sep=";")
    for name in zf.namelist():
        if name.endswith(".zip"):
            with zf.open(name) as inner:
                found = _find_and_read(zipfile.ZipFile(io.BytesIO(inner.read())), filename)
                if found is not None:
                    return found
    return None


def load_student_data(course="por", url=DATA_URL):
    """course: 'por' (Portuguese) or 'mat' (Math)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
    df = _find_and_read(zf, f"student-{course}.csv")
    if df is None:
        raise FileNotFoundError(f"student-{course}.csv not found. Archive has: {zf.namelist()}")
    return df
