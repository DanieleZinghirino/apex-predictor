"""
Pipeline end-to-end di Apex Predictor.
Carica i dati, costruisce le feature, allena il modello campione (XGBoost tunato), lo valuta sul test set 2023-2024 e salva modello + configurazione in models/.

Modello scelto per priorità a precision alta (soglia 0.75).
Vedi README e notebooks/03_model_comparison.ipynb, notebooks/04_hyperparameter_tuning.ipynb per il ragionamento completo.
"""
from xgboost import XGBClassifier

from src.data_loading import load_raw_data, build_working_dataset
from src.features import build_all_features
from src.train import (
    temporal_split, train_champion_model, find_best_threshold,
    evaluate_model, save_model
)

MODEL_THRESHOLD = 0.75

# Iperparametri trovati con RandomizedSearchCV + TimeSeriesSplit
# (vedi notebooks/04_hyperparameter_tuning.ipynb)
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.01,
    "subsample": 0.6,
    "colsample_bytree": 0.8,
    "reg_alpha": 1,
    "reg_lambda": 2,
    "random_state": 42,
    "eval_metric": "logloss",
}

print("1. Caricamento dati grezzi...")
data = load_raw_data()

print("2. Costruzione dataset di lavoro (filtro 2004+, target podium)...")
df = build_working_dataset(data["races"], data["results"], min_year=2004)
print(f"   Righe: {df.shape[0]}")

print("3. Feature engineering...")
df = build_all_features(df, data["circuits"], n_races=10)
print(f"   Righe dopo pulizia NaN: {df.shape[0]}")

print("4. Split temporale train/test (train < 2025, test 2025-2026)...")
train_df, test_df = temporal_split(df)
print(f"   Train: {train_df.shape[0]} righe, Test: {test_df.shape[0]} righe")

print("5. Training del modello campione (XGBoost)...")
model = train_champion_model(train_df)

print("6. Ricerca soglia ottimale...")
best = find_best_threshold(model, test_df)
print(f"   Soglia ottimale: {best['threshold']:.2f}")
print(f"   Precision: {best['precision']:.3f}, Recall: {best['recall']:.3f}, F1: {best['f1']:.3f}")

print("\n7. Valutazione completa alla soglia scelta...")
print(evaluate_model(model, test_df, threshold=best["threshold"]))

print("8. Salvataggio modello...")
save_model(model, threshold=best["threshold"])

print("\nPipeline completata con successo.")