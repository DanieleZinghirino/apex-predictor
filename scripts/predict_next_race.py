"""
Genera la previsione di podio per la prossima gara F1 in calendario, usando lo storico locale aggiornato e la griglia di partenza live.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_loading import load_raw_data, build_working_dataset
from src.jolpica_client import get_next_race_info, get_qualifying_results
from src.live_predict import map_refs_to_ids, build_upcoming_race_features
from src.predict import load_trained_model, predict_podium

print("Recupero informazioni sulla prossima gara...")
race_info = get_next_race_info()

if race_info is None:
    print("Nessuna gara futura trovata in calendario.")
    sys.exit(0)

print(f"Prossima gara: {race_info['race_name']} ({race_info['date']})")

print("Recupero griglia di partenza...")
qualifying = get_qualifying_results(race_info["season"], race_info["round"])

data = load_raw_data()
historical_df = build_working_dataset(data["races"], data["results"], min_year=2004)

circuit_map = dict(zip(data["circuits"]["circuitRef"], data["circuits"]["circuitId"]))
circuit_id = circuit_map.get(race_info["circuit_ref"])

if circuit_id is None:
    print(f"ATTENZIONE: circuito '{race_info['circuit_ref']}' non mappato.")
    sys.exit(1)

if qualifying:
    print("Griglia di qualifica reale disponibile; previsione:")
    qualifying_df = map_refs_to_ids(qualifying, data["drivers"], data["constructors"])
    features_df = build_upcoming_race_features(qualifying_df, historical_df, circuit_id, data["circuits"])
    features_df["is_estimated_grid"] = 0
else:
    print("Qualifiche non ancora disponibili; previsione:")
    from src.live_predict import build_pre_qualifying_features
    features_df = build_pre_qualifying_features(historical_df, circuit_id, int(race_info["season"]), data["circuits"])


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
print(f"\nPrevisione {label} per {race_info['race_name']}:\n")
print(result.to_string(index=False))