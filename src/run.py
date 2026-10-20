"""Punto de entrada.

  python -m src.run collect     # cada hora: realidad + previsiones + calibración + dashboard
                                # (y avanza el histórico pendiente dentro del cupo de la API)
  python -m src.run backfill --start 2025-01-01   # programa el histórico inicial y avanza lo que quepa
  python -m src.run build       # solo regenerar dashboard

La API gratuita de Open-Meteo tiene cupo (~5.000 llamadas/hora, ~10.000/día, ponderadas por
variables y días). El histórico se carga por tramos de 30 días a lo largo de varias ejecuciones,
empezando por los spots que tienen estación con histórico (Tarifa, Bilbao, Santander, Cascais).
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone

from . import obs as O
from .build import build_all
from .calibrate import calibrate_all
from .common import DATA, RateLimited, load_spots, write_json
from .forecast import fetch_snapshot, history_weight, update_fc_history

STATE = DATA / "backfill_state.json"
HOURLY_BUDGET = 1500     # llamadas ponderadas por ejecución dedicadas al histórico
DAILY_BUDGET = 6000      # y por día (deja margen para la recogida normal)
CHUNK_DAYS = 30
# Spots con estación de histórico (METAR) primero: calibran desde el primer día
PRIORITY = ["valdevaqueros", "los_lances", "sopelana", "mundaka", "somo", "guincho", "carcavelos", "caparica",
            "portil", "punta_umbria", "sagres_tonel", "sagres_oeste", "sagres_sur", "praia_grande", "supertubos",
            "baleal", "canos", "el_palmar", "roche", "conil", "zarautz", "oyambre", "ericeira"]


def _now():
    return datetime.now(timezone.utc)


def _load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def _save_state(st: dict):
    write_json(STATE, st)


def _resume_from(spot_id: str, start: str) -> str:
    """Si ya hay histórico descargado (p.ej. una ejecución cortada), seguimos desde donde acabó."""
    p = DATA / "fc_hist" / f"{spot_id}.parquet"
    if not p.exists():
        return start
    import pandas as pd
    t = pd.read_parquet(p, columns=["time"])["time"]
    old = t[t < pd.Timestamp(_now().date() - timedelta(days=5))]
    if old.empty or old.min() > pd.Timestamp(start) + pd.Timedelta(days=2):
        return start
    return (old.max() + pd.Timedelta(hours=1)).date().isoformat()


def backfill_step(budget: float | None = None) -> dict:
    """Avanza el histórico pendiente sin pasarse del cupo. Devuelve qué ha hecho."""
    st = _load_state()
    if not st or st.get("done"):
        return {"status": "nada pendiente"}
    today = _now().date().isoformat()
    spent_today = st.setdefault("spent", {}).get(today, 0.0)
    budget = min(budget or HOURLY_BUDGET, DAILY_BUDGET - spent_today)
    end = date.fromisoformat(st["end"])
    order = sorted(load_spots(), key=lambda s: PRIORITY.index(s["id"]) if s["id"] in PRIORITY else 99)
    used, chunks = 0.0, 0
    try:
        for s in order:
            if s["id"] not in st["cursor"]:
                st["cursor"][s["id"]] = _resume_from(s["id"], st["start"])
            cur = date.fromisoformat(st["cursor"][s["id"]])
            while cur <= end:
                c_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
                w = history_weight((c_end - cur).days + 1)
                if used + w > budget:
                    raise StopIteration
                n = update_fc_history(s, cur, c_end)
                print(f"[backfill] {s['id']} {cur}..{c_end}: {n} filas", flush=True)
                used += w
                chunks += 1
                cur = c_end + timedelta(days=1)
                st["cursor"][s["id"]] = cur.isoformat()
                _save_state(st)
                time.sleep(1)
        st["done"] = True
    except StopIteration:
        pass
    except RateLimited as e:
        print(f"[backfill] cupo de la API agotado, sigo en la próxima ejecución: {e}")
    finally:
        st["spent"][today] = round(spent_today + used, 1)
        st["spent"] = {k: v for k, v in st["spent"].items() if k >= (_now().date() - timedelta(days=3)).isoformat()}
        _save_state(st)
    pend = sum(1 for s in load_spots() if date.fromisoformat(st["cursor"].get(s["id"], st["start"])) <= end)
    return {"chunks": chunks, "weight": round(used), "spots_pendientes": pend, "done": st.get("done", False)}


def collect():
    t0 = time.time()
    now = _now()
    log = {"started": now.isoformat(timespec="minutes")}
    log["obs"] = O.collect_recent()
    today = now.date()
    fc = {}
    # Previsiones: los modelos salen cada 6 h -> snapshot cada 3 h y previsiones pasadas cada 6 h
    do_snap = now.hour % 3 == 0 or not (DATA / "snapshots").exists() or not any((DATA / "snapshots").iterdir())
    do_hist = now.hour % 6 == 0 or not any((DATA / "fc_hist").glob("*.parquet"))
    try:
        for s in load_spots():
            try:
                if do_hist:
                    fc[s["id"]] = update_fc_history(s, today - timedelta(days=3), today + timedelta(days=1))
                if do_snap or not (DATA / "snapshots" / f"{s['id']}.parquet").exists():
                    fetch_snapshot(s)
            except RateLimited:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[fc {s['id']}] falló: {e}")
                fc[s["id"]] = f"error: {e}"
    except RateLimited as e:
        print(f"[fc] cupo de la API agotado: {e}")
        log["rate_limited"] = True
    log["fc_rows"] = fc
    if not log.get("rate_limited"):
        log["backfill"] = backfill_step()
    calibrate_all()
    summ = build_all()
    log["spots_built"] = len(summ["spots"])
    log["seconds"] = round(time.time() - t0)
    write_json(DATA / "last_run.json", log)
    print(json.dumps(log, ensure_ascii=False, indent=1, default=str))


def backfill(start: date):
    print("Meteostat:", O.backfill_meteostat(start.year))
    st = _load_state()
    if not st or st.get("start") != start.isoformat():
        st = {"start": start.isoformat(), "end": (_now().date() - timedelta(days=1)).isoformat(), "cursor": {},
              "spent": {}, "done": False}
        _save_state(st)
    print("Histórico:", backfill_step(budget=HOURLY_BUDGET * 2))
    collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "backfill", "build"])
    ap.add_argument("--start", default="2025-01-01")
    a = ap.parse_args()
    if a.cmd == "collect":
        collect()
    elif a.cmd == "backfill":
        backfill(date.fromisoformat(a.start))
    else:
        calibrate_all()
        build_all()


if __name__ == "__main__":
    main()
