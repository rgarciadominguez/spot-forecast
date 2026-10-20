"""Calibración: cuánto se equivoca cada modelo en cada spot y cómo construir la horquilla.

Para cada spot con estación cercana:
1. Une previsión pasada (por modelo y antelación) con lo medido en la estación (horas de luz).
2. Pesa los modelos según su error a cada antelación (el que más acierta en ese spot pesa más).
3. Corrige el sesgo según dirección y fuerza (p.ej. "con levante se queda 3 kn corto").
4. La horquilla (P10–P90) sale de la distribución real de errores; se abre si los modelos discrepan.
5. Mide el acierto con datos que el ajuste no ha visto (últimos 30 días).

Sin estación cercana: pesos iguales, sin corrección y horquilla genérica (media de los spots calibrados).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import DATA, LEADS, MODEL_LABEL, WIND_MODELS, clean_nans, load_spots, sector, write_json
from .forecast import load_fc_history
from .obs import load_obs

SPEED_BINS = [0, 8, 14, 20, 26, 80]
SPEED_LABELS = ["<8", "8-14", "14-20", "20-26", "26+"]
NEAR_KM = 15.0          # estación representativa del spot
MIN_CELL = 40           # mínimo de horas para fiarnos de una celda
HOLDOUT_DAYS = 30
DAY_HOURS = range(8, 21)  # horas locales que cuentan (navegables)


def _speed_bin(x):
    return pd.cut(x, SPEED_BINS, labels=SPEED_LABELS, right=False)


def _circ_wmean(dirs: pd.DataFrame, weights: dict) -> pd.Series:
    s = c = 0
    tot = 0
    for m, w in weights.items():
        if m not in dirs:
            continue
        r = np.deg2rad(dirs[m])
        ok = dirs[m].notna()
        s = s + np.where(ok, np.sin(r) * w, 0)
        c = c + np.where(ok, np.cos(r) * w, 0)
    return pd.Series(np.rad2deg(np.arctan2(s, c)) % 360, index=dirs.index)


def blend(wide_speed: pd.DataFrame, weights: dict) -> tuple[pd.Series, pd.Series]:
    """Media ponderada de modelos (renormaliza si falta alguno) y dispersión entre modelos."""
    cols = [m for m in weights if m in wide_speed]
    if not cols:
        return pd.Series(np.nan, index=wide_speed.index), pd.Series(np.nan, index=wide_speed.index)
    W = np.array([weights[m] for m in cols])
    X = wide_speed[cols].to_numpy(dtype=float)
    mask = ~np.isnan(X)
    num = np.nansum(X * W, axis=1)
    den = (mask * W).sum(axis=1)
    central = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    spread = np.nanstd(np.where(mask, X, np.nan), axis=1)
    return pd.Series(central, index=wide_speed.index), pd.Series(spread, index=wide_speed.index)


def pick_station(spot_id: str, stations: dict):
    """Primera estación de la lista con datos suficientes."""
    for st in stations.get(spot_id, []):
        obs = load_obs(st["net"], st["id"])
        if len(obs) >= 24:
            return st, obs
    return None, None


def _fit(train: pd.DataFrame) -> dict:
    """Ajusta pesos, correcciones y cuantiles a partir de filas unidas (una por time×model×lead)."""
    params = {"weights": {}, "corr": {}, "q": {}, "spread_med": {}, "gust_ratio": {}, "lead_stats": {}}
    for lead in LEADS:
        d = train[train["lead"] == lead]
        if d.empty:
            continue
        # Error por modelo
        mae = d.assign(ae=(d["wspd"] - d["obs"]).abs()).groupby("model")["ae"].agg(["mean", "count"])
        mae = mae[mae["count"] >= 100]
        if mae.empty:
            w = {m: 1.0 for m in WIND_MODELS}
        else:
            inv = 1 / np.maximum(mae["mean"], 0.5) ** 2
            w = (inv / inv.sum()).to_dict()
        params["weights"][lead] = w
        # Serie mezclada
        wide = d.pivot_table(index="time", columns="model", values="wspd")
        wdir = d.pivot_table(index="time", columns="model", values="wdir")
        wgust = d.pivot_table(index="time", columns="model", values="gust")
        central, spread = blend(wide, w)
        obs = d.groupby("time")[["obs", "obs_gust"]].first()
        b = pd.DataFrame({"fc": central, "spread": spread, "dir": _circ_wmean(wdir, w)}).join(obs).dropna(subset=["fc", "obs"])
        if b.empty:
            continue
        b["sector"] = b["dir"].map(sector)
        b["bin"] = _speed_bin(b["fc"]).astype(str)
        b["res"] = b["obs"] - b["fc"]
        lead_med = float(b["res"].median())
        # Corrección por sector y fuerza, encogida hacia la de la antelación si hay pocos datos
        corr = {"_": lead_med}
        for (sec, bn), g in b.groupby(["sector", "bin"]):
            n = len(g)
            k = n / (n + 50)
            corr[f"{sec}|{bn}"] = k * float(g["res"].median()) + (1 - k) * lead_med
        params["corr"][lead] = corr
        b["fc_c"] = b["fc"] + b.apply(lambda r: corr.get(f"{r['sector']}|{r['bin']}", lead_med), axis=1)
        b["res_c"] = b["obs"] - b["fc_c"]
        # Cuantiles del error ya corregido, por fuerza
        q = {"_": [float(b["res_c"].quantile(0.1)), float(b["res_c"].quantile(0.9))]}
        for bn, g in b.groupby("bin"):
            if len(g) >= MIN_CELL:
                q[bn] = [float(g["res_c"].quantile(0.1)), float(g["res_c"].quantile(0.9))]
        params["q"][lead] = q
        params["spread_med"][lead] = float(max(b["spread"].median(), 0.5))
        # Rachas: relación racha medida / racha prevista
        gb = wgust.mean(axis=1).rename("fc_gust")
        gg = b.join(gb).dropna(subset=["obs_gust", "fc_gust"])
        gg = gg[gg["fc_gust"] > 5]
        params["gust_ratio"][lead] = float((gg["obs_gust"] / gg["fc_gust"]).median()) if len(gg) >= MIN_CELL else 1.0
        # Estadísticas para la pantalla "¿cuánto me fío?"
        per_model = {}
        for m, g in d.groupby("model"):
            per_model[MODEL_LABEL.get(m, m)] = {"mae": float((g["wspd"] - g["obs"]).abs().mean()),
                                                "bias": float((g["obs"] - g["wspd"]).mean()), "n": int(len(g))}
        sec_bias = {s: {"bias": float(g["res"].median()), "n": int(len(g))}
                    for s, g in b.groupby("sector") if len(g) >= MIN_CELL}
        params["lead_stats"][lead] = {"models": per_model, "mae_blend_raw": float(b["res"].abs().mean()),
                                      "mae_blend": float(b["res_c"].abs().mean()), "n": int(len(b)),
                                      "sector_bias": sec_bias}
    return params


def apply(params: dict, lead: int, models_speed: pd.DataFrame, models_dir: pd.DataFrame,
          models_gust: pd.DataFrame) -> pd.DataFrame:
    """Aplica la calibración a una previsión (filas = horas, columnas = modelos)."""
    lead_key = lead if lead in params.get("weights", {}) else None
    w = params["weights"].get(lead_key) if lead_key is not None else None
    if not w:
        w = {m: 1.0 for m in WIND_MODELS}
    central, spread = blend(models_speed, w)
    d = _circ_wmean(models_dir, w)
    sec = d.map(sector)
    bins = _speed_bin(central).astype(str)
    corr = params.get("corr", {}).get(lead_key, {"_": 0.0}) if lead_key is not None else {"_": 0.0}
    c = pd.Series([corr.get(f"{s}|{b}", corr.get("_", 0.0)) for s, b in zip(sec, bins)], index=central.index)
    p50 = (central + c).clip(lower=0)
    q = params.get("q", {}).get(lead_key) or params.get("generic_q", {}).get(lead, [-4.0, 4.0])
    if isinstance(q, dict):
        lo = pd.Series([q.get(b, q["_"])[0] for b in bins], index=central.index)
        hi = pd.Series([q.get(b, q["_"])[1] for b in bins], index=central.index)
    else:
        lo = pd.Series(q[0], index=central.index)
        hi = pd.Series(q[1], index=central.index)
    smed = params.get("spread_med", {}).get(lead_key, 2.0) if lead_key is not None else 2.0
    f = (spread / smed).clip(0.8, 1.6).fillna(1.0)
    p10 = (p50 + lo * f).clip(lower=0)
    p90 = (p50 + hi * f).clip(lower=0)
    gr = params.get("gust_ratio", {}).get(lead_key, 1.0) if lead_key is not None else 1.0
    gust = models_gust.mean(axis=1) * gr
    return pd.DataFrame({"p10": p10, "p50": p50, "p90": p90, "gust": gust, "wdir": d, "spread": spread,
                         "raw": central})


def _joined(spot_id: str, obs: pd.DataFrame) -> pd.DataFrame:
    fc = load_fc_history(spot_id)
    if fc.empty or obs.empty:
        return pd.DataFrame()
    o = obs.rename(columns={"wspd": "obs", "gust": "obs_gust", "wdir": "obs_dir"})[["time", "obs", "obs_gust", "obs_dir"]]
    j = fc.merge(o, on="time", how="inner")
    local_h = j["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Madrid").dt.hour
    return j[local_h.isin(DAY_HOURS)]


def _coverage(params: dict, test: pd.DataFrame) -> dict:
    out = {}
    for lead in LEADS:
        d = test[test["lead"] == lead]
        if d.empty:
            continue
        sp = d.pivot_table(index="time", columns="model", values="wspd")
        di = d.pivot_table(index="time", columns="model", values="wdir")
        gu = d.pivot_table(index="time", columns="model", values="gust")
        r = apply(params, lead, sp, di, gu).join(d.groupby("time")["obs"].first()).dropna(subset=["obs", "p50"])
        if len(r) < 24:
            continue
        inside = ((r["obs"] >= r["p10"]) & (r["obs"] <= r["p90"])).mean()
        out[lead] = {"coverage": float(inside), "mae": float((r["obs"] - r["p50"]).abs().mean()),
                     "width": float((r["p90"] - r["p10"]).median()), "n": int(len(r))}
    return out


def calibrate_all() -> dict:
    stations = json.loads((DATA / "stations.json").read_text(encoding="utf-8")) if (DATA / "stations.json").exists() else {}
    result = {}
    generic_pool = {l: [] for l in LEADS}
    for spot in load_spots():
        st, obs = pick_station(spot["id"], stations)
        entry = {"station": st, "quality": "sin estación", "params": None}
        if st is not None:
            j = _joined(spot["id"], obs)
            if len(j) >= 200:
                cutoff = j["time"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
                train, test = j[j["time"] < cutoff], j[j["time"] >= cutoff]
                honest = train["time"].nunique() >= 300
                params = _fit(train if honest else j)
                params["coverage"] = _coverage(params, test if honest else j)
                params["coverage_mode"] = "fuera de muestra (últimos 30 días)" if honest else "dentro de muestra (pocos datos aún)"
                params["hours"] = int(j["time"].nunique())
                params["from"] = str(j["time"].min())[:10]
                entry["params"] = params
                entry["quality"] = "orilla" if (st["km"] <= NEAR_KM and st["net"] != "meteostat") else \
                    ("cercana" if st["km"] <= NEAR_KM else "referencia lejana")
                if st["km"] <= NEAR_KM:
                    for l, q in params["q"].items():
                        generic_pool[l].append(q["_"])
            else:
                entry["quality"] = "acumulando datos"
                entry["hours"] = int(j["time"].nunique()) if not j.empty else 0
        result[spot["id"]] = entry
    generic = {l: [float(np.mean([q[0] for q in v])), float(np.mean([q[1] for q in v]))] if v else [-4.0, 4.0]
               for l, v in generic_pool.items()}
    for e in result.values():
        if e["params"] is None or e["quality"] == "referencia lejana":
            # Spot sin estación representativa: no corregimos sesgo local, horquilla genérica
            base = e["params"] or {}
            e["params"] = {"weights": base.get("weights", {}), "corr": {}, "q": {}, "spread_med": base.get("spread_med", {}),
                           "gust_ratio": {}, "generic_q": generic, "lead_stats": base.get("lead_stats", {}),
                           "coverage": base.get("coverage", {}), "coverage_mode": base.get("coverage_mode", ""),
                           "hours": base.get("hours", e.get("hours", 0)), "from": base.get("from", "")}
        else:
            e["params"]["generic_q"] = generic
    write_json(DATA / "calibration.json", clean_nans(result))
    return result
