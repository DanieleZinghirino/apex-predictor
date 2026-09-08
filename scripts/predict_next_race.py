"""
Genera la previsione di podio per la prossima gara F1 in calendario, usando lo storico locale aggiornato e la griglia di partenza live.
"""
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading import load_raw_data, build_working_dataset, PROJECT_ROOT
from src.jolpica_client import get_next_race_info, get_qualifying_results
from src.live_predict import (
    map_refs_to_ids,
    build_upcoming_race_features,
    build_pre_qualifying_features,
    compute_live_qualifying_gaps,
)
from src.predict import load_trained_model, predict_podium

print("Recupero informazioni sulla prossima gara...")
race_info = get_next_race_info()

if race_info is None:
    print("Nessuna gara futura trovata in calendario.")
    sys.exit(0)

print(f"Prossima gara: {race_info['race_name']} ({race_info['date']})")

print("Recupero griglia di partenza...")
qualifying = get_qualifying_results(race_info["season"], race_info["round"])
if qualifying:
    qualifying = compute_live_qualifying_gaps(qualifying)

data = load_raw_data()
historical_df = build_working_dataset(data["races"], data["results"], min_year=2004)

circuit_map = dict(zip(data["circuits"]["circuitRef"], data["circuits"]["circuitId"]))
circuit_id = circuit_map.get(race_info["circuit_ref"])

if circuit_id is None:
    # Circuito debuttante: crealo automaticamente usando le coordinate fornite da Jolpica, invece di fallire
    if race_info.get("circuit_lat") and race_info.get("circuit_lng"):
        new_id = int(data["circuits"]["circuitId"].max()) + 1
        new_row = {
            "circuitId": new_id,
            "circuitRef": race_info["circuit_ref"],
            "name": race_info["circuit_name"],
            "location": race_info.get("circuit_country", ""),
            "country": race_info.get("circuit_country", ""),
            "lat": float(race_info["circuit_lat"]),
            "lng": float(race_info["circuit_lng"]),
        }
        data["circuits"] = pd.concat([data["circuits"], pd.DataFrame([new_row])], ignore_index=True)
        data["circuits"].to_csv(os.path.join(PROJECT_ROOT, "data", "raw", "circuits.csv"), index=False)
        circuit_id = new_id
        print(f"Circuito nuovo rilevato e creato automaticamente: {race_info['circuit_ref']} (circuitId={new_id})")

        # Tentativo best-effort di recuperare lunghezza/curve da Wikipedia
        characteristics_path = os.path.join(PROJECT_ROOT, "data", "reference", "circuit_characteristics.csv")
        characteristics_df = pd.read_csv(characteristics_path)

        if race_info["circuit_ref"] not in characteristics_df["circuit_ref"].values:
            from src.circuit_scraper import scrape_circuit_characteristics

            wiki_url = f"https://en.wikipedia.org/wiki/{race_info['circuit_name'].replace(' ', '_')}"
            scraped = scrape_circuit_characteristics(wiki_url)

            if scraped["length_km"] and scraped["num_corners"]:
                new_char_row = {
                    "circuit_ref": race_info["circuit_ref"],
                    "length_km": scraped["length_km"],
                    "num_corners": scraped["num_corners"],
                    "direction": "clockwise",
                    "altitude_m": 0,
                    "downforce_level": "medium",
                }
                characteristics_df = pd.concat([characteristics_df, pd.DataFrame([new_char_row])], ignore_index=True)
                characteristics_df.to_csv(characteristics_path, index=False)
                print(f"  Caratteristiche circuito recuperate automaticamente da Wikipedia: "
                      f"{scraped['length_km']}km, {scraped['num_corners']} curve")
            else:
                print(f"  Impossibile estrarre automaticamente le caratteristiche del circuito da Wikipedia "
                      f"(formato pagina non riconosciuto) — verrà usato il valore medio globale come fallback.")
    else:
        print(f"ATTENZIONE: circuito '{race_info['circuit_ref']}' non mappato e senza coordinate, impossibile procedere.")
        sys.exit(1)

if qualifying:
    qualifying_df = map_refs_to_ids(qualifying, data["drivers"], data["constructors"])
    features_df = build_upcoming_race_features(
        qualifying_df, historical_df, circuit_id, data["circuits"], race_date=race_info["date"]
    )
    features_df["is_estimated_grid"] = 0
else:
    features_df = build_pre_qualifying_features(
        historical_df, circuit_id, int(race_info["season"]), data["circuits"], race_date=race_info["date"]
    )

print("Caricamento modello e generazione previsioni...")
model, threshold = load_trained_model()
predictions = predict_podium(model, threshold, features_df)

predictions = predictions.merge(
    data["drivers"][["driverId", "surname"]], on="driverId", how="left"
)

result = predictions[["surname", "grid", "podium_probability", "podium_predicted", "is_estimated_grid"]] \
    .sort_values("podium_probability", ascending=False)
result.columns = ["Pilota", "Griglia", "Probabilità podio", "Predetto", "Griglia stimata"]


label = "DEFINITIVA" if qualifying else "ANTICIPATA (griglia stimata)"
# Il meteo è uguale per tutti i piloti della stessa gara, lo mostriamo una sola volta come intestazione, non ripetuto su ogni riga
if "race_max_temp_c" in features_df.columns and not features_df.empty:
    temp = features_df["race_max_temp_c"].iloc[0]
    precip = features_df["race_precipitation_mm"].iloc[0]
    meteo_status = "pioggia probabile" if precip > 1.0 else "asciutto"
    print(f"\nMeteo previsto: {temp:.1f}°C, {precip:.1f}mm di pioggia ({meteo_status})")

print(f"\nPrevisione {label} per {race_info['race_name']}:\n")
print(result.to_string(index=False))