"""
Recupera i dati meteo storici per tutte le gare dal 2004 in poi e li salva in data/reference/race_weather.csv, anche se qui via API, non manualmente.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from src.data_loading import load_raw_data, PROJECT_ROOT
from src.weather_client import get_race_weather

data = load_raw_data()
races = data["races"][data["races"]["year"] >= 2004]
circuits = data["circuits"].set_index("circuitId")[["lat", "lng"]]

output_path = os.path.join(PROJECT_ROOT, "data", "reference", "race_weather.csv")

# Idempotenza: se il file esiste già, riprendi da dove eri arrivato invece di rifare tutte le chiamate da zero
if os.path.exists(output_path):
    existing = pd.read_csv(output_path)
    done_race_ids = set(existing["raceId"])
else:
    existing = pd.DataFrame(columns=["raceId", "max_temp_c", "precipitation_mm"])
    done_race_ids = set()

results = [existing] if not existing.empty else []

for _, race in races.iterrows():
    if race["raceId"] in done_race_ids:
        continue

    circuit = circuits.loc[race["circuitId"]]
    weather = get_race_weather(circuit["lat"], circuit["lng"], race["date"])

    if weather:
        results.append(pd.DataFrame([{
            "raceId": race["raceId"],
            "max_temp_c": weather["max_temp_c"],
            "precipitation_mm": weather["precipitation_mm"],
        }]))
        print(f"  {race['name']} ({race['date']}): {weather}")
    else:
        print(f"  {race['name']} ({race['date']}): dato non disponibile, salto")

    # Salvataggio incrementale
    pd.concat(results, ignore_index=True).to_csv(output_path, index=False)

print(f"\nCompletato. Dati salvati in {output_path}")