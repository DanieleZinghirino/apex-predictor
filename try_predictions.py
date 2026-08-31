"""
Script dimostrativo: genera previsioni su una gara già nel dataset (usando il modello salvato) e le confronta con il risultato reale.
"""

from src.data_loading import load_raw_data, build_working_dataset
from src.features import build_all_features
from src.predict import load_trained_model, predict_podium

print("Caricamento dati e ricostruzione feature...")
data = load_raw_data()
df = build_working_dataset(data["races"], data["results"], min_year=2004)
df = build_all_features(df,10)

print("Caricamento modello addestrato...")
model, threshold = load_trained_model()

# Prendiamo l'ultima gara disponibile nel dataset
last_race_id = df["raceId"].max()
race_data = df[df["raceId"] == last_race_id].copy()

print(f"\nGara selezionata: raceId {last_race_id}, data {race_data['date'].iloc[0].date()}")

# Generiamo le previsioni
race_data = predict_podium(model, threshold, race_data)

# Aggiungiamo il nome del pilota per leggibilità (drivers ha driverId + surname)
drivers_names = data["drivers"][["driverId", "surname"]]
race_data = race_data.merge(drivers_names, on="driverId", how="left")

# Ordiniamo per probabilità decrescente e mostriamo le colonne rilevanti
result = race_data[[
    "surname", "grid", "podium_probability", "podium_predicted", "podium"
]].sort_values("podium_probability", ascending=False)

result.columns = ["Pilota", "Griglia", "Probabilità podio", "Predetto", "Reale"]
print("\n" + result.to_string(index=False))