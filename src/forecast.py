"""Previsiones desde Open-Meteo (mismos modelos que usan Windguru/Windy).

- fetch_fc_history(): lo que decía cada modelo 0,1,2,3,5 días antes, para cada hora (Previous Runs API).
  Es la base para medir cuánto se equivoca cada modelo y a qué antelación.
- fetch_snapshot(): la previsión completa más reciente de un spot (viento por modelo, cielo,
  temperatura, lluvia, oleaje por modelo, marea, temperatura del agua, gradiente de presión).
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

# Puntos para el gradiente de presión del Estrecho (levante/poniente)
STRAIT_W = (36.53, -6.30)   # Cádiz
STRAIT_E = (36.72, -4.42)   # Málaga


def _suffix(lead: int) -> str:
    return "" if lead == 0 else f"_previous_day{lead}"


def fetch_fc_history(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Previsiones pasadas por antelación. Devuelve formato largo: time, model, lead, wspd, gust, wdir."""
    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=29), end)
        hourly = [v + _suffix(l) for l in LEADS for v in WIND_VARS]
        js = get_json(PREV_URL, params={
            "latitude": lat, "longitude": lon, "hourly": ",".join(hourly), "models": ",".join(WIND_MODELS),
            "start_date": cur.isoformat(), "end_date": chunk_end.isoformat(), "wind_speed_unit": "kn",
            "timezone": "GMT",
        }, timeout=120)
        h = js.get("hourly", {})
        t = pd.to_datetime(h.get("time", []))
        for m in WIND_MODELS:
            for l in LEADS:
                cols = {}
                for v, short in WIND_VARS.items():
                    key = f"{v}{_suffix(l)}_{m}"
                    cols[short] = h.get(key)
                if cols["wspd"] is None:
                    continue
                df = pd.DataFrame({"time": t, **{k: (v if v is not None else [np.nan] * len(t)) for k, v in cols.items()}})
                df["model"], df["lead"] = m, l
                frames.append(df.dropna(subset=["wspd"]))
        cur = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["time", "model", "lead", "wspd", "gust", "wdir"])
    out = pd.concat(frames, ignore_index=True)
    out["lead"] = out["lead"].astype("int8")
    for c in ("wspd", "gust", "wdir"):
        out[c] = out[c].astype("float32")
    return out[["time", "model", "lead", "wspd", "gust", "wdir"]]


def request_weight(days: int, n_vars: int) -> float:
    """Coste aproximado de una petición en 'llamadas' de la API gratuita de Open-Meteo."""
    return max(1.0, n_vars / 10) * max(1.0, days / 14)


def history_weight(days: int) -> float:
    return request_weight(days, len(LEADS) * len(WIND_VARS) * len(WIND_MODELS))


def update_fc_history(spot: dict, start: date, end: date) -> int:
    df = fetch_fc_history(spot["lat"], spot["lon"], start, end)
    upsert_parquet(FC_DIR / f"{spot['id']}.parquet", df, ["time", "model", "lead"])
    return len(df)


def load_fc_history(spot_id: str) -> pd.DataFrame:
    p = FC_DIR / f"{spot_id}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["time", "model", "lead", "wspd", "gust", "wdir"])


def _frame(js: dict) -> pd.DataFrame:
    h = js.get("hourly", {})
    df = pd.DataFrame(h)
    if "time" in df:
        df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_snapshot(spot: dict) -> pd.DataFrame:
    """Previsión horaria completa (UTC) desde hace 3 días hasta +7 días."""
    lat, lon = spot["lat"], spot["lon"]
    common = {"timezone": "GMT", "past_days": 3, "forecast_days": 7}

    # 1) Viento por modelo
    w = _frame(get_json(FC_URL, params={**common, "latitude": lat, "longitude": lon,
                                         "hourly": ",".join(WIND_VARS), "models": ",".join(WIND_MODELS),
                                         "wind_speed_unit": "kn"}))
    ren = {}
    for v, short in WIND_VARS.items():
        for m in WIND_MODELS:
            ren[f"{v}_{m}"] = f"{short}__{m}"
    w = w.rename(columns=ren)

    # 2) Cielo, temperatura, lluvia, luz (mejor modelo disponible)
    a = _frame(get_json(FC_URL, params={**common, "latitude": lat, "longitude": lon,
                                         "hourly": "cloud_cover,temperature_2m,precipitation,is_day,pressure_msl"}))

    # 3) Temperatura tierra adentro (~12 km) -> motor del térmico
    ilat, ilon = offset_point(lat, lon, (spot["facing"] + 180) % 360, 12)
    inland = _frame(get_json(FC_URL, params={**common, "latitude": ilat, "longitude": ilon,
                                              "hourly": "temperature_2m"})).rename(
        columns={"temperature_2m": "t_inland"})

    # 4) Oleaje por modelo, en un punto ~4 km mar adentro
    mlat, mlon = offset_point(lat, lon, spot["facing"], 4)
    wave_vars = ["wave_height", "wave_period", "wave_direction", "swell_wave_height",
                 "swell_wave_peak_period", "swell_wave_direction", "wind_wave_height"]
    try:
        wv = _frame(get_json(MARINE_URL, params={**common, "latitude": mlat, "longitude": mlon,
                                                  "hourly": ",".join(wave_vars), "models": ",".join(WAVE_MODELS),
                                                  "cell_selection": "sea"}))
        wv = wv.rename(columns={f"{v}_{m}": f"{v}__{m}" for v in wave_vars for m in WAVE_MODELS})
    except Exception as e:  # noqa: BLE001
        print(f"[marine {spot['id']}] oleaje falló: {e}")
        wv = pd.DataFrame({"time": w["time"]})

    # 5) Marea + temperatura del agua (modelo por defecto)
    try:
        tide = _frame(get_json(MARINE_URL, params={**common, "latitude": mlat, "longitude": mlon,
                                                    "hourly": "sea_level_height_msl,sea_surface_temperature",
                                                    "cell_selection": "sea"}))
    except Exception as e:  # noqa: BLE001
        print(f"[marine {spot['id']}] marea falló: {e}")
        tide = pd.DataFrame({"time": w["time"]})

    df = w.merge(a, on="time", how="left").merge(inland, on="time", how="left") \
        .merge(wv, on="time", how="left").merge(tide, on="time", how="left")

    # 6) Gradiente de presión del Estrecho (hPa): >0 empuja levante, <0 poniente
    if spot.get("zone") in ("Estrecho", "Cádiz"):
        try:
            js = get_json(FC_URL, params={**common, "latitude": f"{STRAIT_W[0]},{STRAIT_E[0]}",
                                          "longitude": f"{STRAIT_W[1]},{STRAIT_E[1]}", "hourly": "pressure_msl"})
            west, east = _frame(js[0]), _frame(js[1])
            g = pd.DataFrame({"time": west["time"], "grad_estrecho": east["pressure_msl"] - west["pressure_msl"]})
            df = df.merge(g, on="time", how="left")
        except Exception as e:  # noqa: BLE001
            print(f"[gradiente] falló: {e}")

    df.to_parquet(SNAP_DIR / f"{spot['id']}.parquet", index=False)
    return df


def load_snapshot(spot_id: str) -> pd.DataFrame | None:
    p = SNAP_DIR / f"{spot_id}.parquet"
    return pd.read_parquet(p) if p.exists() else None
