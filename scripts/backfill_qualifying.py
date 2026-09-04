"""
Backfill mirato: scarica le qualifiche mancanti per le gare 2025-2026 già presenti in races.csv ma non ancora in qualifying.csv, perché update_data.py 
inizialmente non le scaricava
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data_loading import load_raw_data, PROJECT_ROOT
from src.jolpica_client import get_qualifying_results

data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
data = load_raw_data()

races_to_check = data["races"][data["races"]["year"] >= 2025]
qualifying_df = data.get("qualifying")

if qualifying_df is None:
    qualifying_df = pd.read_csv(os.path.join(data_dir, "qualifying.csv"))

existing_race_ids_in_qualifying = set(qualifying_df["raceId"])

driver_map = dict(zip(data["drivers"]["driverRef"], data["drivers"]["driverId"]))
constructor_map = dict(zip(data["constructors"]["constructorRef"], data["constructors"]["constructorId"]))

for _, race in races_to_check.iterrows():
    if race["raceId"] in existing_race_ids_in_qualifying:
        continue  # già presente, salta (idempotenza)

    print(f"Scaricamento qualifiche: {race['year']} round {race['round']} — {race['name']}")
    qualifying_results = get_qualifying_results(race["year"], race["round"])

    if not qualifying_results:
        print("  Nessun dato disponibile, salto.")
        continue

    new_qualify_id = int(qualifying_df["qualifyId"].max()) + 1 if not qualifying_df.empty else 1

    for q in qualifying_results:
        driver_id = driver_map.get(q["driver_ref"])
        constructor_id = constructor_map.get(q["constructor_ref"])

        if driver_id is None or constructor_id is None:
            print(f"  ATTENZIONE: {q['driver_ref']} non mappato, riga saltata")
            continue

        new_row = {
            "qualifyId": new_qualify_id,
            "raceId": race["raceId"],
            "driverId": driver_id,
            "constructorId": constructor_id,
            "position": q["grid_position"],
            "q1": q["q1"],
            "q2": q["q2"],
            "q3": q["q3"],
        }
        qualifying_df = pd.concat([qualifying_df, pd.DataFrame([new_row])], ignore_index=True)
        new_qualify_id += 1

    # Salvataggio incrementale
    qualifying_df.to_csv(os.path.join(data_dir, "qualifying.csv"), index=False)
    print(f"  Aggiunte {len(qualifying_results)} righe di qualifica")

print("\nCompletato.")