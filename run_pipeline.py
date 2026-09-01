"""
Pipeline end-to-end di Apex Predictor.
Carica i dati, costruisce le feature, allena il modello campione (XGBoost tunato), lo valuta sul test set 2023-2024 e salva modello + configurazione in models/.

Modello scelto per priorità a precision alta (soglia 0.75).
Vedi README e notebooks/03_model_comparison.ipynb, notebooks/04_hyperparameter_tuning.ipynb per il ragionamento completo.
"""
from xgboost import XGBClassifier

from src.data_loading import load_raw_data, build_working_dataset
from src.features import build_all_features
from src.train import temporal_split, evaluate_model, save_model, FEATURE_COLS

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
df = build_all_features(df, n_races=10)
print(f"   Righe dopo pulizia NaN: {df.shape[0]}")

print("4. Split temporale train/test...")
train_df, test_df = temporal_split(df, test_start_year=2023)
print(f"   Train: {train_df.shape[0]} righe, Test: {test_df.shape[0]} righe")

print("5. Training del modello (XGBoost, iperparametri ottimizzati)...")
X_train, y_train = train_df[FEATURE_COLS], train_df["podium"]
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)
model.fit(X_train, y_train)

print("6. Valutazione...")
report = evaluate_model(model, test_df, threshold=MODEL_THRESHOLD)
print(report)

print("7. Salvataggio modello...")
save_model(model, threshold=MODEL_THRESHOLD, output_dir="models")

print("\nPipeline completata con successo.")