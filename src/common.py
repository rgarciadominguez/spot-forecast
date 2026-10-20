"""Utilidades comunes: config, HTTP con reintentos, geometría, almacenamiento."""
from __future__ import annotations

import gzip
import io
import json
import math
import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCS = ROOT / "docs"
OBS_DIR = DATA / "obs"
FC_DIR = DATA / "fc_hist"          # previsiones pasadas por antelación (para calibrar)
SNAP_DIR = DATA / "snapshots"      # última previsión completa (para el dashboard)
for d in (OBS_DIR, FC_DIR, SNAP_DIR, DOCS / "data"):
    d.mkdir(parents=True, exist_ok=True)

KN_PER_KMH = 1 / 1.852
KN_PER_MS = 1.943844

# Modelos de viento (los que hay debajo de Windguru/Windy)
WIND_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "meteofrance_seamless", "ukmo_seamless"]
MODEL_LABEL = {
    "ecmwf_ifs025": "ECMWF",
    "gfs_seamless": "GFS",
    "icon_seamless": "ICON",
    "meteofrance_seamless": "AROME/ARPEGE",
    "ukmo_seamless": "UKMO",
}
WAVE_MODELS = ["ecmwf_wam025", "meteofrance_wave", "ncep_gfswave025"]
# Antelaciones que evaluamos (días antes). 0 = la pasada más reciente.
LEADS = [0, 1, 2, 3, 5]

UA = {"User-Agent": "spot-forecast/1.0 (github.com/rgarciadominguez/spot-forecast)"}


def load_spots() -> list[dict]:
    return yaml.safe_load((ROOT / "spots.yaml").read_text(encoding="utf-8"))["spots"]


def load_rules() -> dict:
    return yaml.safe_load((ROOT / "rules.yaml").read_text(encoding="utf-8"))


def http_get(url: str, params: dict | None = None, *, tries: int = 4, timeout: int = 60,
             headers: dict | None = None) -> requests.Response:
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={**UA, **(headers or {})})
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"{r.status_code} {r.text[:200]}")
                time.sleep(5 * (i + 1) ** 2)
                continue
            return r
        except requests.RequestException as e:  # red caída, timeout...
            last = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"GET falló tras {tries} intentos: {url} -> {last}")


def get_json(url: str, params: dict | None = None, **kw):
    r = http_get(url, params, **kw)
    r.raise_for_status()
    return r.json()


def get_gz_text(url: str) -> str | None:
    r = http_get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return gzip.decompress(r.content).decode("utf-8")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def offset_point(lat, lon, bearing_deg, km):
    """Punto a `km` en dirección `bearing` (para tomar el oleaje un poco mar adentro)."""
    R = 6371.0
    b = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    d = km / R
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return round(math.degrees(p2), 4), round(math.degrees(l2), 4)


def ang_diff(a, b):
    """Diferencia angular mínima (0-180)."""
    d = abs((a - b) % 360)
    return min(d, 360 - d)


def wind_relation(wdir, facing) -> str:
    """onshore / cross-on / cross / cross-off / offshore respecto a la playa."""
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)):
        return ""
    d = ang_diff(wdir, facing)  # 0 = viene del mar de frente
    if d <= 30:
        return "onshore"
    if d <= 67.5:
        return "cross-on"
    if d <= 112.5:
        return "cross"
    if d <= 150:
        return "cross-off"
    return "offshore"


SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SECTORS_ES = {"N": "N", "NE": "NE", "E": "E (levante)", "SE": "SE", "S": "S", "SW": "SW", "W": "W (poniente)", "NW": "NW"}


def sector(wdir) -> str:
    if wdir is None or (isinstance(wdir, float) and math.isnan(wdir)):
        return ""
    return SECTORS[int(((wdir % 360) + 22.5) // 45) % 8]


def upsert_parquet(path: Path, new: pd.DataFrame, keys: list[str]):
    """Añade filas nuevas y sustituye las que coinciden en `keys`."""
    if new is None or new.empty:
        return
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new], ignore_index=True)
    else:
        df = new
    df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)
    df.to_parquet(path, index=False)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=_nan_safe),
                    encoding="utf-8")


def _nan_safe(o):
    return None


def clean_nans(obj):
    """Sustituye NaN/inf por None recursivamente (JSON válido)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else round(obj, 2)
    if isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_nans(v) for v in obj]
    if hasattr(obj, "item"):  # numpy scalars
        return clean_nans(obj.item())
    return obj


def env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None
