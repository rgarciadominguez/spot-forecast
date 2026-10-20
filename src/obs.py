"""Recolección de la REALIDAD: estaciones de viento en orilla o lo más cerca posible.

Fuentes:
- Flecha Extreme (El Portil): estación propia, cada 5 min, últimas 24 h.
- IPMA (Portugal): todas las estaciones, horario, últimas 24 h. Sin clave.
- AEMET OpenData (España): todas las estaciones, horario, últimas 24 h. Necesita AEMET_API_KEY.
- Meteostat: METAR/SYNOP de aeropuertos/puertos con histórico (para el backtest inicial).

Todo se guarda en nudos, por hora UTC, en data/obs/<red>_<id>.parquet
"""
from __future__ import annotations

import io
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .common import (KN_PER_KMH, KN_PER_MS, OBS_DIR, DATA, env, get_gz_text, get_json, haversine_km,
                     http_get, load_spots, upsert_parquet, write_json)

MADRID = "Europe/Madrid"

# ---------------------------------------------------------------- estaciones fijas
FLECHA = {"net": "flecha", "id": "portil", "name": "Flecha Extreme (El Portil)", "lat": 37.2075, "lon": -7.0450,
          "kind": "orilla"}

# Meteostat: solo estaciones con observación real (METAR/ISD), no relleno de modelo
METEOSTAT = [
    {"id": "08495", "name": "Gibraltar (aeropuerto)", "lat": 36.15, "lon": -5.35},
    {"id": "08025", "name": "Bilbao aeropuerto", "lat": 43.30, "lon": -2.91},
    {"id": "08021", "name": "Santander Parayas", "lat": 43.43, "lon": -3.82},
    {"id": "LPCS0", "name": "Cascais Tires", "lat": 38.73, "lon": -9.35},
    {"id": "08451", "name": "Jerez aeropuerto", "lat": 36.75, "lon": -6.07},
]
METEOSTAT_BAD_SOURCES = ("forecast", "mosmix", "model", "poi", "era5", "interp")

IPMA_DIR = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315, 9: 0}


def _circmean(deg: pd.Series) -> float:
    d = deg.dropna()
    if d.empty:
        return np.nan
    r = np.deg2rad(d.values)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def _hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega filas sub-horarias (time UTC) a horario: media viento, racha máx, dirección media."""
    if df.empty:
        return df
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_localize(None).dt.floor("h")
    g = df.groupby("time")
    out = pd.DataFrame({
        "wspd": g["wspd"].mean(),
        "gust": g["gust"].max() if "gust" in df else np.nan,
        "wdir": g["wdir"].apply(_circmean) if "wdir" in df else np.nan,
        "n": g["wspd"].count(),
    }).reset_index()
    return out


def _save(net: str, sid: str, hourly: pd.DataFrame):
    if hourly is None or hourly.empty:
        return 0
    hourly = hourly.dropna(subset=["wspd"])
    upsert_parquet(OBS_DIR / f"{net}_{sid}.parquet", hourly[["time", "wspd", "gust", "wdir", "n"]], ["time"])
    return len(hourly)


# ---------------------------------------------------------------- Flecha Extreme
def fetch_flecha() -> int:
    base = "https://www.flechaextreme.com"
    now_local = pd.Timestamp.now(tz=MADRID)
    rows = []
    # Histórico de 24 h cada 5 min (media + racha, sin dirección)
    try:
        hist = get_json(f"{base}/get_history.php")
        # viene del más antiguo al más reciente; solo trae HH:MM -> reconstruimos la fecha hacia atrás
        day = now_local.normalize()
        prev_minutes = None
        stamped = []
        for item in reversed(hist):
            hh, mm = map(int, item["hora"].split(":"))
            minutes = hh * 60 + mm
            if prev_minutes is None:
                if minutes > now_local.hour * 60 + now_local.minute + 10:
                    day -= pd.Timedelta(days=1)
            elif minutes > prev_minutes:
                day -= pd.Timedelta(days=1)
            prev_minutes = minutes
            ts = (day + pd.Timedelta(minutes=minutes)).tz_convert("UTC") if day.tzinfo else day
            stamped.append({"time": ts, "wspd": float(item["viento_media"]), "gust": float(item["viento_raf"]),
                            "wdir": np.nan})
        rows += stamped
    except Exception as e:  # noqa: BLE001
        print(f"[flecha] historial falló: {e}")
    # Lectura actual con dirección (realtimegauges.txt, JSON con coma decimal)
    try:
        r = http_get(f"{base}/meteo/realtimegauges.txt")
        import json
        g = json.loads(r.text)
        num = lambda k: float(str(g.get(k, "nan")).replace(",", "."))
        rows.append({"time": pd.Timestamp.now(tz="UTC"), "wspd": num("wspeed"), "gust": num("wgust"),
                     "wdir": num("avgbearing")})
    except Exception as e:  # noqa: BLE001
        print(f"[flecha] realtime falló: {e}")
    if not rows:
        return 0
    raw = pd.DataFrame(rows)
    raw["time"] = pd.to_datetime(raw["time"], utc=True).dt.tz_localize(None)
    # Guardamos crudo (5 min) para no perder dirección entre ejecuciones
    upsert_parquet(OBS_DIR / "flecha_portil_raw.parquet", raw, ["time"])
    allraw = pd.read_parquet(OBS_DIR / "flecha_portil_raw.parquet")
    allraw = allraw[allraw["time"] >= allraw["time"].max() - pd.Timedelta(days=3)]
    return _save("flecha", "portil", _hourly(allraw))


# ---------------------------------------------------------------- IPMA (Portugal)
def ipma_stations() -> list[dict]:
    js = get_json("https://api.ipma.pt/open-data/observation/meteorology/stations/stations.json")
    out = []
    for f in js:
        lon, lat = f["geometry"]["coordinates"]
        out.append({"net": "ipma", "id": str(f["properties"]["idEstacao"]),
                    "name": f["properties"]["localEstacao"], "lat": lat, "lon": lon, "kind": "estación"})
    return out


def fetch_ipma(wanted: set[str]) -> int:
    if not wanted:
        return 0
    js = get_json("https://api.ipma.pt/open-data/observation/meteorology/stations/observations.json")
    per = {sid: [] for sid in wanted}
    for t, stations in js.items():
        for sid in wanted:
            o = (stations or {}).get(sid)
            if not o:
                continue
            v = o.get("intensidadeVento", -99)
            if v is None or v < 0:
                continue
            d = IPMA_DIR.get(o.get("idDireccVento"), np.nan)
            per[sid].append({"time": pd.Timestamp(t, tz="UTC"), "wspd": v * KN_PER_MS, "gust": np.nan, "wdir": d})
    n = 0
    for sid, rows in per.items():
        if rows:
            n += _save("ipma", sid, _hourly(pd.DataFrame(rows)))
    return n


# ---------------------------------------------------------------- AEMET (España)
def _aemet_all() -> list[dict] | None:
    key = env("AEMET_API_KEY")
    if not key:
        return None
    meta = get_json("https://opendata.aemet.es/opendata/api/observacion/convencional/todas",
                    params={"api_key": key}, timeout=90)
    url = meta.get("datos")
    if not url:
        print(f"[aemet] sin datos: {meta}")
        return None
    r = http_get(url, timeout=120)
    r.encoding = "latin-1"
    import json
    return json.loads(r.text)


def aemet_stations(records: list[dict] | None) -> list[dict]:
    if not records:
        return []
    seen = {}
    for x in records:
        if x.get("vv") is None:
            continue
        seen[x["idema"]] = {"net": "aemet", "id": x["idema"], "name": x.get("ubi", x["idema"]).title(),
                            "lat": x["lat"], "lon": x["lon"], "kind": "estación"}
    return list(seen.values())


def fetch_aemet(records: list[dict] | None, wanted: set[str]) -> int:
    if not records or not wanted:
        return 0
    per = {}
    for x in records:
        if x["idema"] not in wanted or x.get("vv") is None:
            continue
        per.setdefault(x["idema"], []).append({
            "time": pd.Timestamp(x["fint"]).tz_localize("UTC") if pd.Timestamp(x["fint"]).tzinfo is None
            else pd.Timestamp(x["fint"]),
            "wspd": float(x["vv"]) * KN_PER_MS,
            "gust": float(x["vmax"]) * KN_PER_MS if x.get("vmax") is not None else np.nan,
            "wdir": float(x["dv"]) if x.get("dv") is not None else np.nan,
        })
    return sum(_save("aemet", sid, _hourly(pd.DataFrame(rows))) for sid, rows in per.items())


# ---------------------------------------------------------------- Meteostat (histórico)
def fetch_meteostat(sid: str, years: list[int]) -> int:
    frames = []
    for y in years:
        try:
            txt = get_gz_text(f"https://data.meteostat.net/hourly/{y}/{sid}.csv.gz")
        except Exception as e:  # noqa: BLE001
            print(f"[meteostat {sid} {y}] falló: {e}", flush=True)
            continue
        if not txt:
            continue
        df = pd.read_csv(io.StringIO(txt))
        if "wspd" not in df:
            continue
        src = df.get("wspd_source", pd.Series([""] * len(df))).fillna("").astype(str)
        ok = ~src.str.contains("|".join(METEOSTAT_BAD_SOURCES))
        df = df[ok & df["wspd"].notna()]
        if df.empty:
            continue
        t = pd.to_datetime(df[["year", "month", "day", "hour"]])
        frames.append(pd.DataFrame({
            "time": t,
            "wspd": df["wspd"] * KN_PER_KMH,
            "gust": df["wpgt"] * KN_PER_KMH if "wpgt" in df else np.nan,
            "wdir": df["wdir"],
            "n": 1,
        }))
    if not frames:
        return 0
    out = pd.concat(frames)
    out = out[out["time"] <= pd.Timestamp.utcnow().tz_localize(None)]
    return _save("meteostat", sid, out)


# ---------------------------------------------------------------- asignación spot -> estación
def assign_stations(aemet_records=None, max_km: float = 25.0) -> dict:
    """Para cada spot, lista de estaciones cercanas (orden por distancia). Se guarda en data/stations.json."""
    cands = [FLECHA] + [{"net": "meteostat", "kind": "aeropuerto/puerto", **m} for m in METEOSTAT]
    try:
        cands += ipma_stations()
    except Exception as e:  # noqa: BLE001
        print(f"[ipma] estaciones falló: {e}")
    cands += aemet_stations(aemet_records)
    result = {}
    for s in load_spots():
        near = []
        for c in cands:
            d = haversine_km(s["lat"], s["lon"], c["lat"], c["lon"])
            # Las de aeropuerto lejanas sirven de referencia hasta 40 km
            lim = 40.0 if c["net"] == "meteostat" else max_km
            if d <= lim:
                near.append({**c, "km": round(d, 1)})
        near.sort(key=lambda x: (x["km"] + (8 if x["net"] == "meteostat" else 0)))
        result[s["id"]] = near[:4]
    write_json(DATA / "stations.json", result)
    return result


def collect_recent() -> dict:
    """Ejecución horaria: trae las últimas 24 h de todas las redes."""
    aemet = None
    try:
        aemet = _aemet_all()
    except Exception as e:  # noqa: BLE001
        print(f"[aemet] falló: {e}")
    stations = assign_stations(aemet)
    wanted = {"ipma": set(), "aemet": set()}
    for lst in stations.values():
        for c in lst:
            if c["net"] in wanted:
                wanted[c["net"]].add(c["id"])
    stats = {}
    for name, fn in [("flecha", fetch_flecha),
                     ("ipma", lambda: fetch_ipma(wanted["ipma"])),
                     ("aemet", lambda: fetch_aemet(aemet, wanted["aemet"]))]:
        try:
            stats[name] = fn()
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] falló: {e}")
            stats[name] = f"error: {e}"
    # Meteostat: refresco del año en curso cada 6 h (los METAR llegan con retraso)
    now = datetime.now(timezone.utc)
    y = now.year
    for m in (METEOSTAT if now.hour % 6 == 1 else []):
        try:
            stats[f"meteostat_{m['id']}"] = fetch_meteostat(m["id"], [y])
        except Exception as e:  # noqa: BLE001
            print(f"[meteostat {m['id']}] falló: {e}")
    return stats


def backfill_meteostat(start_year: int) -> dict:
    y = datetime.now(timezone.utc).year
    out = {}
    for m in METEOSTAT:
        out[m["id"]] = fetch_meteostat(m["id"], list(range(start_year, y + 1)))
        print(f"[meteostat] {m['name']}: {out[m['id']]} horas", flush=True)
    return out


def load_obs(net: str, sid: str) -> pd.DataFrame:
    p = OBS_DIR / f"{net}_{sid}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["time", "wspd", "gust", "wdir", "n"])
    return pd.read_parquet(p)
