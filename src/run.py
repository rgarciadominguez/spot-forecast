"""Punto de entrada.

  python -m src.run collect     # cada hora: realidad + previsiones recientes + calibración + dashboard
  python -m src.run backfill --start 2025-01-01   # histórico inicial (una vez)
  python -m src.run build       # solo regenerar dashboard
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone

from . import obs as O
from .build import build_all
from .calibrate import calibrate_all
from .common import DATA, load_spots, write_json
from .forecast import fetch_snapshot, update_fc_history


def collect(full_snapshot: bool = True):
    t0 = time.time()
    log = {"started": datetime.now(timezone.utc).isoformat(timespec="minutes")}
    log["obs"] = O.collect_recent()
    today = datetime.now(timezone.utc).date()
    fc = {}
    for s in load_spots():
        try:
            # Últimos 3 días de previsiones pasadas (rellena antelaciones 0..5 de horas ya ocurridas)
            fc[s["id"]] = update_fc_history(s, today - timedelta(days=3), today + timedelta(days=1))
            if full_snapshot:
                fetch_snapshot(s)
        except Exception as e:  # noqa: BLE001
            print(f"[fc {s['id']}] falló: {e}")
            fc[s["id"]] = f"error: {e}"
    log["fc_rows"] = fc
    calibrate_all()
    summ = build_all()
    log["spots_built"] = len(summ["spots"])
    log["seconds"] = round(time.time() - t0)
    write_json(DATA / "last_run.json", log)
    print(json.dumps(log, ensure_ascii=False, indent=1, default=str))


def backfill(start: date):
    print("Meteostat:", O.backfill_meteostat(start.year))
    O.assign_stations(O._aemet_all() if O.env("AEMET_API_KEY") else None)
    end = datetime.now(timezone.utc).date()
    for s in load_spots():
        try:
            n = update_fc_history(s, start, end)
            print(f"{s['id']}: {n} filas de previsión")
        except Exception as e:  # noqa: BLE001
            print(f"[backfill {s['id']}] falló: {e}")
        time.sleep(2)  # amable con la API gratuita
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
