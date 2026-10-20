"""Previsiones desde Open-Meteo (mismos modelos que usan Windguru/Windy).

Todas las peticiones van agrupadas por varios spots a la vez (coordenadas separadas por comas):
pocas peticiones = menos bloqueos de la API gratuita desde servidores compartidos.

- update_fc_history(): lo que decía cada modelo 0,1,2,3,5 días antes, para cada hora (Previous Runs API).
- fetch_snapshots(): la previsión más reciente (viento por modelo, cielo, temperatura, lluvia, oleaje por
  modelo, marea, temperatura del agua y gradiente de presión del Estrecho).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from .common import (FC_DIR, LEADS, SNAP_DIR, WAVE_MODELS, WIND_MODELS, get_json, offset_point,
                     upsert_parquet)

PREV_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FC_URL = "https://api.open-meteo.com/v1/forecast"
MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WIND_VARS = {"wind_speed_10m": "wspd", "wind_gusts_10m": "gust", "wind_direction_10m": "wdir"}
GROUP = 6  # spots por petición

# Puntos para el gradiente de presión del Estrecho (levante/poniente)
STRAIT_W = (36.53, -6.30)   # Cádiz
STRAIT_E = (36.72, -4.42)   # Málaga


def _suffix(lead: int) -> str:
    return "" if lead == 0 else f"_previous_day{lead}"


def _multi(url: str, points: list[tuple[float, float]], params: dict, timeout: int = 120) -> list[dict]:
    js = get_json(url, params={**params, "latitude": ",".join(str(p[0]) for p in points),
                               "longitude": ",".join(str(p[1]) for p in points)}, timeout=timeout)
    return js if isinstance(js, list) else [js]


def _frame(js: dict) -> pd.DataFrame:
    df = pd.DataFrame(js.get("hourly", {}))
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    return df


def _groups(spots):
    return [spots[i:i + GROUP] for i in range(0, len(spots), GROUP)]


# ------------------------------------------------------------------ histórico por antelación
def _long_history(js: dict) -> pd.DataFrame:
    h = js.get("hourly", {})
    t = pd.to_datetime(h.get("time", []))
    frames = []
    for m in WIND_MODELS:
        for l in LEADS:
            cols = {short: h.get(f"{v}{_suffix(l)}_{m}") for v, short in WIND_VARS.items()}
            if cols["wspd"] is None:
                continue
            df = pd.DataFrame({"time": t, **{k: (v if v is not None else [np.nan] * len(t)) for k, v in cols.items()}})
            df["model"], df["lead"] = m, l
            frames.append(df.dropna(subset=["wspd"]))
    if not frames:
        return pd.DataFrame(columns=["time", "model", "lead", "wspd", "gust", "wdir"])
    out = pd.concat(frames, ignore_index=True)
    out["lead"] = out["lead"].astype("int8")
    for c in ("wspd", "gust", "wdir"):
        out[c] = out[c].astype("float32")
    return out[["time", "model", "lead", "wspd", "gust", "wdir"]]


def update_fc_history(spots: list[dict], start: date, end: date) -> dict:
    """Descarga y guarda previsiones pasadas para varios spots (tramos de ≤30 días)."""
    hourly = ",".join(v + _suffix(l) for l in LEADS for v in WIND_VARS)
    rows = {s["id"]: 0 for s in spots}
    for grp in _groups(spots):
        cur = start
        while cur <= end:
            c_end = min(cur + timedelta(days=29), end)
            res = _multi(PREV_URL, [(s["lat"], s["lon"]) for s in grp], {
                "hourly": hourly, "models": ",".join(WIND_MODELS), "start_date": cur.isoformat(),
                "end_date": c_end.isoformat(), "wind_speed_unit": "kn", "timezone": "GMT"}, timeout=180)
            for s, js in zip(grp, res):
                df = _long_history(js)
                upsert_parquet(FC_DIR / f"{s['id']}.parquet", df, ["time", "model", "lead"])
                rows[s["id"]] += len(df)
            cur = c_end + timedelta(days=1)
    return rows


def request_weight(days: int, n_vars: int, n_locations: int = 1) -> float:
    """Coste aproximado en 'llamadas' de la API gratuita de Open-Meteo."""
    return max(1.0, n_vars / 10) * max(1.0, days / 14) * n_locations


def history_weight(days: int, n_locations: int = 1) -> float:
    return request_weight(days, len(LEADS) * len(WIND_VARS) * len(WIND_MODELS), n_locations)


def load_fc_history(spot_id: str) -> pd.DataFrame:
    p = FC_DIR / f"{spot_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["time", "model", "lead", "wspd", "gust", "wdir"])


# ------------------------------------------------------------------ previsión actual
def fetch_snapshots(spots: list[dict]) -> int:
    """Previsión horaria completa (UTC) desde hace 3 días hasta +7 días para todos los spots."""
    common = {"timezone": "GMT", "past_days": 3, "forecast_days": 7}
    wave_vars = ["wave_height", "wave_period", "wave_direction", "swell_wave_height",
                 "swell_wave_peak_period", "swell_wave_direction", "wind_wave_height"]
    ok = 0
    for grp in _groups(spots):
        pts = [(s["lat"], s["lon"]) for s in grp]
        inland = [offset_point(s["lat"], s["lon"], (s["facing"] + 180) % 360, 12) for s in grp]
        sea = [offset_point(s["lat"], s["lon"], s["facing"], 4) for s in grp]
        wind = _multi(FC_URL, pts, {**common, "hourly": ",".join(WIND_VARS), "models": ",".join(WIND_MODELS),
                                    "wind_speed_unit": "kn"})
        atmo = _multi(FC_URL, pts, {**common, "hourly": "cloud_cover,temperature_2m,precipitation,is_day,pressure_msl"})
        inl = _multi(FC_URL, inland, {**common, "hourly": "temperature_2m"})
        try:
            waves = _multi(MARINE_URL, sea, {**common, "hourly": ",".join(wave_vars), "models": ",".join(WAVE_MODELS),
                                             "cell_selection": "sea"})
        except Exception as e:  # noqa: BLE001
            print(f"[marine] oleaje falló: {e}", flush=True)
            waves = [{}] * len(grp)
        try:
            tide = _multi(MARINE_URL, sea, {**common, "hourly": "sea_level_height_msl,sea_surface_temperature",
                                            "cell_selection": "sea"})
        except Exception as e:  # noqa: BLE001
            print(f"[marine] marea falló: {e}", flush=True)
            tide = [{}] * len(grp)
        for i, s in enumerate(grp):
            w = _frame(wind[i]).rename(columns={f"{v}_{m}": f"{short}__{m}" for v, short in WIND_VARS.items()
                                                for m in WIND_MODELS})
            df = w.merge(_frame(atmo[i]), on="time", how="left")
            df = df.merge(_frame(inl[i]).rename(columns={"temperature_2m": "t_inland"}), on="time", how="left")
            wv = _frame(waves[i]).rename(columns={f"{v}_{m}": f"{v}__{m}" for v in wave_vars for m in WAVE_MODELS})
            if "time" in wv:
                df = df.merge(wv, on="time", how="left")
            td = _frame(tide[i])
            if "time" in td:
                df = df.merge(td, on="time", how="left")
            df.to_parquet(SNAP_DIR / f"{s['id']}.parquet", index=False)
            ok += 1
    # Gradiente de presión del Estrecho (hPa): >0 empuja levante, <0 poniente
    strait = [s for s in spots if s.get("zone") in ("Estrecho", "Cádiz")]
    if strait:
        try:
            west, east = (_frame(x) for x in _multi(FC_URL, [STRAIT_W, STRAIT_E], {**common, "hourly": "pressure_msl"}))
            g = pd.DataFrame({"time": west["time"], "grad_estrecho": east["pressure_msl"] - west["pressure_msl"]})
            for s in strait:
                p = SNAP_DIR / f"{s['id']}.parquet"
                if p.exists():
                    d = pd.read_parquet(p).drop(columns=["grad_estrecho"], errors="ignore").merge(g, on="time", how="left")
                    d.to_parquet(p, index=False)
        except Exception as e:  # noqa: BLE001
            print(f"[gradiente] falló: {e}", flush=True)
    return ok


def load_snapshot(spot_id: str) -> pd.DataFrame | None:
    p = SNAP_DIR / f"{spot_id}.parquet"
    return pd.read_parquet(p) if p.exists() else None
