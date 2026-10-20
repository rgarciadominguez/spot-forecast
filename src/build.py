"""Genera los JSON que lee el dashboard (docs/data/)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .calibrate import apply, pick_station
from .common import (DATA, DOCS, LEADS, MODEL_LABEL, SECTORS_ES, WAVE_MODELS, WIND_MODELS, clean_nans,
                     load_rules, load_spots, sector, wind_relation, write_json)
from .forecast import load_fc_history, load_snapshot

TZ = "Europe/Madrid"
NAV_HOURS = range(8, 21)
LEVEL_ORDER = ["", "bueno", "muy_bueno", "excepcional"]


def _norm_keys(d):
    """JSON convierte claves int en str; las devolvemos a int donde son antelaciones."""
    if isinstance(d, dict):
        return {(int(k) if isinstance(k, str) and k.isdigit() else k): _norm_keys(v) for k, v in d.items()}
    return d


def load_calibration() -> dict:
    p = DATA / "calibration.json"
    return _norm_keys(json.loads(p.read_text(encoding="utf-8"))) if p.exists() else {}


def _lead_for(hours_ahead: float) -> int:
    days = hours_ahead / 24
    for l in LEADS:
        if days < l + 1:
            return l
    return LEADS[-1]


def _wide(df: pd.DataFrame, var: str, models) -> pd.DataFrame:
    cols = {m: f"{var}__{m}" for m in models}
    return pd.DataFrame({m: df[c] for m, c in cols.items() if c in df}, index=df.index)


def _circ_median(df: pd.DataFrame) -> pd.Series:
    r = np.deg2rad(df.to_numpy(dtype=float))
    s, c = np.nanmean(np.sin(r), axis=1), np.nanmean(np.cos(r), axis=1)
    return pd.Series(np.rad2deg(np.arctan2(s, c)) % 360, index=df.index)


def _level(hours: int, rule: dict, extra_ok: bool) -> str:
    lv = rule["levels"]
    if hours >= lv["excepcional"] and extra_ok:
        return "excepcional"
    if hours >= lv["muy_bueno"]:
        return "muy_bueno"
    if hours >= lv["bueno"]:
        return "bueno"
    return ""


def _windows(mask: pd.Series) -> str:
    """'13–18h' con la racha de horas buenas más larga."""
    best, cur, start, best_start = 0, 0, None, None
    for h, ok in mask.items():
        if ok:
            cur += 1
            if cur == 1:
                start = h
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    if not best:
        return ""
    return f"{best_start:02d}–{best_start + best:02d}h"


def sport_day(day: pd.DataFrame, sport: str, rule: dict) -> dict:
    nav = day[day["hour"].isin(NAV_HOURS) & (day["is_day"].fillna(1) > 0)].set_index("hour")
    if nav.empty:
        return {"hours": 0, "sure": 0, "level": ""}
    if sport == "surf":
        hs, per = nav["hs"], nav["tp"]
        wind_ok = nav["rel"].isin(["offshore", "cross-off", "cross"]) | (nav["p50"] <= rule["wind_max_onshore"])
        ok = hs.between(rule["wave_min"], rule["wave_max"]) & (per >= rule["period_min"]) & wind_ok
        sure = hs.ge(rule["wave_min"]) & nav["hs_lo"].ge(rule["wave_min"] * 0.8) & ok
        extra = bool((per[ok] >= rule["excepcional_min_period"]).any())
        summ = f"{hs[ok].min():.1f}–{hs[ok].max():.1f} m · {per[ok].median():.0f} s" if ok.any() else ""
    else:
        bad_rel = nav["rel"].isin(rule["avoid_relation"])
        ok = nav["p50"].between(rule["wind_min"], rule["wind_max"]) & ~bad_rel
        sure = ok & (nav["p10"] >= rule["wind_min"])
        extra = bool((nav["p50"][ok] >= rule["excepcional_min_wind"]).sum() >= 3)
        summ = f"{nav['p10'][ok].min():.0f}–{nav['p90'][ok].max():.0f} kn" if ok.any() else ""
    n = int(ok.sum())
    return {"hours": n, "sure": int(sure.sum()), "level": _level(n, rule, extra), "window": _windows(ok),
            "range": summ}


def build_spot(spot: dict, calib: dict, stations: dict, rules: dict) -> dict | None:
    snap = load_snapshot(spot["id"])
    if snap is None or snap.empty:
        return None
    entry = calib.get(spot["id"], {})
    params = entry.get("params") or {"weights": {}, "generic_q": {l: [-4.0, 4.0] for l in LEADS}}
    snap = snap.sort_values("time").reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    snap = snap[(snap["time"] >= now - pd.Timedelta(hours=48))].reset_index(drop=True)

    sp, di, gu = (_wide(snap, v, WIND_MODELS) for v in ("wspd", "wdir", "gust"))
    out = pd.DataFrame({"time": snap["time"]})
    out[["p10", "p50", "p90", "gust", "wdir", "spread", "raw"]] = np.nan
    ahead = (snap["time"] - now).dt.total_seconds() / 3600
    future = ahead >= 0
    for lead in LEADS:
        idx = future & (ahead.map(_lead_for) == lead)
        if idx.any():
            r = apply(params, lead, sp[idx], di[idx], gu[idx])
            out.loc[idx, r.columns] = r.values

    # Pasado: lo que se preveía AYER (D-1) para comparar con lo medido
    past = ~future
    fch = load_fc_history(spot["id"])
    if past.any() and not fch.empty:
        d1 = fch[(fch["lead"] == 1) & (fch["time"].isin(snap.loc[past, "time"]))]
        if not d1.empty:
            p_sp = d1.pivot_table(index="time", columns="model", values="wspd")
            p_di = d1.pivot_table(index="time", columns="model", values="wdir")
            p_gu = d1.pivot_table(index="time", columns="model", values="gust")
            r = apply(params, 1, p_sp, p_di, p_gu)
            r = r.reindex(snap.loc[past, "time"]).reset_index(drop=True)
            r.index = snap.index[past]
            out.loc[past, r.columns] = r.values
    # Si falta histórico D-1, usamos la pasada actual (lead 0) como referencia
    miss = past & out["p50"].isna()
    if miss.any():
        r = apply(params, 0, sp[miss], di[miss], gu[miss])
        out.loc[miss, r.columns] = r.values

    # Realidad medida
    st, obs = pick_station(spot["id"], stations)
    if obs is not None:
        o = obs.set_index("time")
        out["obs"] = out["time"].map(o["wspd"])
        out["obs_gust"] = out["time"].map(o["gust"])
        out["obs_dir"] = out["time"].map(o["wdir"])
    else:
        out["obs"] = out["obs_gust"] = out["obs_dir"] = np.nan

    # Cielo, temperaturas, marea, oleaje
    for c in ("cloud_cover", "temperature_2m", "precipitation", "is_day", "t_inland", "sea_surface_temperature",
              "sea_level_height_msl", "grad_estrecho"):
        out[c] = snap[c] if c in snap else np.nan
    hs = _wide(snap, "wave_height", WAVE_MODELS)
    out["hs"] = hs.median(axis=1)
    out["hs_lo"], out["hs_hi"] = hs.min(axis=1), hs.max(axis=1)
    out["tp"] = _wide(snap, "wave_period", WAVE_MODELS).median(axis=1)
    out["swell"] = _wide(snap, "swell_wave_height", WAVE_MODELS).median(axis=1)
    out["swell_tp"] = _wide(snap, "swell_wave_peak_period", WAVE_MODELS).median(axis=1)
    out["swell_dir"] = _circ_median(_wide(snap, "swell_wave_direction", WAVE_MODELS)) if not hs.empty else np.nan
    out["wave_dir"] = _circ_median(_wide(snap, "wave_direction", WAVE_MODELS)) if not hs.empty else np.nan
    out["thermal"] = out["t_inland"] - out["sea_surface_temperature"]
    out["rel"] = out["wdir"].map(lambda d: wind_relation(d, spot["facing"]))
    out["sector"] = out["wdir"].map(sector)

    loc = out["time"].dt.tz_localize("UTC").dt.tz_convert(TZ)
    out["date"] = loc.dt.strftime("%Y-%m-%d")
    out["hour"] = loc.dt.hour

    # Resumen por día y deporte (solo días futuros)
    today = pd.Timestamp.now(tz=TZ).strftime("%Y-%m-%d")
    days = []
    for d, g in out[out["date"] >= today].groupby("date"):
        rec = {"date": d}
        for s in spot["sports"]:
            rec[s] = sport_day(g, s, rules[s])
        hrs = g[g["hour"].isin(NAV_HOURS)]
        rec["wind"] = {"p10": float(hrs["p10"].min()) if len(hrs) else None,
                       "p90": float(hrs["p90"].max()) if len(hrs) else None,
                       "sector": hrs["sector"].mode().iat[0] if len(hrs) and hrs["sector"].ne("").any() else ""}
        rec["hs"] = float(hrs["hs"].max()) if len(hrs) and hrs["hs"].notna().any() else None
        rec["lead"] = int(_lead_for(((pd.Timestamp(d) + pd.Timedelta(hours=14)).tz_localize(TZ).tz_convert("UTC")
                                     .tz_localize(None) - now).total_seconds() / 3600))
        days.append(rec)

    models_out = {MODEL_LABEL[m]: sp[m].round(1).tolist() for m in sp.columns}
    hours = {
        "t": loc.dt.strftime("%Y-%m-%dT%H:00").tolist(),
        "now_index": int(future.idxmax()) if future.any() else len(out),
        **{c: out[c].round(2).tolist() for c in ["p10", "p50", "p90", "gust", "wdir", "spread", "obs", "obs_gust",
                                                    "obs_dir", "cloud_cover", "temperature_2m", "precipitation",
                                                    "is_day", "thermal", "sea_level_height_msl",
                                                    "sea_surface_temperature", "grad_estrecho", "hs", "hs_lo",
                                                    "hs_hi", "tp", "swell", "swell_tp", "swell_dir", "wave_dir"]},
        "rel": out["rel"].tolist(),
        "models": models_out,
    }
    # Fiabilidad
    p = params
    rel_stats = {
        "coverage": p.get("coverage", {}),
        "coverage_mode": p.get("coverage_mode", ""),
        "hours": p.get("hours", 0),
        "from": p.get("from", ""),
        "weights": {l: {MODEL_LABEL.get(m, m): round(w, 3) for m, w in ws.items()} for l, ws in p.get("weights", {}).items()},
        "lead_stats": p.get("lead_stats", {}),
    }
    spot_out = {
        "id": spot["id"], "name": spot["name"], "zone": spot["zone"], "facing": spot["facing"],
        "sports": spot["sports"], "lat": spot["lat"], "lon": spot["lon"],
        "station": st, "stations": stations.get(spot["id"], []),
        "quality": entry.get("quality", "sin estación"),
        "days": days, "hours": hours, "reliability": rel_stats,
    }
    return clean_nans(spot_out)


def build_all() -> dict:
    calib = load_calibration()
    stations = json.loads((DATA / "stations.json").read_text(encoding="utf-8")) if (DATA / "stations.json").exists() else {}
    rules = load_rules()
    summary = {"generated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "spots": [],
               "sectors": SECTORS_ES}
    for spot in load_spots():
        try:
            s = build_spot(spot, calib, stations, rules)
        except Exception as e:  # noqa: BLE001
            print(f"[build {spot['id']}] falló: {e}")
            import traceback; traceback.print_exc()
            s = None
        if not s:
            continue
        write_json(DOCS / "data" / f"spot_{spot['id']}.json", s)
        cov = s["reliability"]["coverage"].get(1) or s["reliability"]["coverage"].get("1") or {}
        summary["spots"].append({k: s[k] for k in ("id", "name", "zone", "sports", "quality", "days", "facing")} |
                                {"station": (s["station"] or {}).get("name"), "station_km": (s["station"] or {}).get("km"),
                                 "coverage_d1": cov.get("coverage"), "mae_d1": cov.get("mae")})
    write_json(DOCS / "data" / "summary.json", clean_nans(summary))
    return summary
