"""
Script di verifica: testa l'intera pipeline usando le funzioni di src/, dal caricamento dati fino al modello addestrato e salvato.
"""
from src.data_loading import load_raw_data, build_working_dataset
from src.features import build_all_features
from src.train import temporal_split, train_model, evaluate_model, save_model

print("1. Caricamento dati grezzi...")
data = load_raw_data(data_dir="data/raw")

print("2. Costruzione dataset di lavoro (filtro 2004+, target podium)...")
df = build_working_dataset(data["races"], data["results"], min_year=2004)
print(f"   Righe: {df.shape[0]}")

print("3. Feature engineering...")
df = build_all_features(df, n_races=10)
print(f"   Righe dopo pulizia NaN: {df.shape[0]}")

print("4. Split temporale train/test...")
train_df, test_df = temporal_split(df, test_start_year=2023)
print(f"   Train: {train_df.shape[0]} righe, Test: {test_df.shape[0]} righe")

print("5. Training del modello...")
model = train_model(train_df, n_estimators=200, max_depth=8)

print("6. Valutazione...")
report = evaluate_model(model, test_df, threshold=0.6)
print(report)

print("7. Salvataggio modello...")
save_model(model, threshold=0.6, output_dir="models")

print("\nPipeline completata con successo.")