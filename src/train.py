"""
Training e valutazione del modello
"""
import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_COL = [
    "grid", "driver_recent_points_avg", "driver_recent_position_avg",
    "constructor_reliability", "driver_circuit_avg_position",
    "circuit_overtaking_index", "circuit_avg_speed_history",
    "circuit_num_corners",
    "driver_standing_position", "constructor_standing_position",
    "teammate_position_gap", "qualifying_gap_seconds",
    "race_max_temp_c", "race_precipitation_mm",
]
# Iperparametri del modello campione, trovati con RandomizedSearchCV +TimeSeriesSplit (vedi notebooks/04_hyperparameter_tuning.ipynb)
XGB_CHAMPION_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.01,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "eval_metric": "logloss",
}

def temporal_split(df, test_start_year=2025):
    """
    Divide il dataset in train e test rispettando l'ordine temporale: tutto ciò che precede test_start_year va in train, il resto in test
    
    Parametri:
        df: DataFrame con feature già costruite, con colonna 'year'
        test_start_year: prima stagione da includere nel test set
        
    Ritorna:
        Tupla contenente i due set: (train_df, test_df)"""
    train_df = df[df["year"] < test_start_year].copy()
    test_df = df[df["year"] >= test_start_year].copy()

    return train_df, test_df


def train_model(train_df, n_estimators=200, max_depth=8, random_state=42):
    """
    Allena il modello Random Forest

    Parametri:
        train_df: DataFrame di training, con FEATURE_COL e 'podium'
        n_estimators, max_depth, random_state: iperparametri del modello

    Ritorna:
        Il modello RandomForestClassifier addestrato
    """
    X_train = train_df[FEATURE_COL]
    y_train = train_df["podium"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model

def train_champion_model(train_df, params=None, target_col="podium"):
    """
    Allena il modello campione (XGBoost) su un target specifico, 'podium' (default, top 3) o 'winner' (vittoria della gara)

    Parametri:
        train_df: DataFrame di training, con FEATURE_COL e target_col
        params: iperparametri XGBoost; None usa XGB_CHAMPION_PARAMS target_col: 'podium' o 'winner'

    Ritorna:
        Il modello XGBClassifier addestrato
    """
    if params is None:
        params = XGB_CHAMPION_PARAMS

    X_train = train_df[FEATURE_COL]
    y_train = train_df[target_col]

    # Con 'winner' lo sbilanciamento è molto più estremo (1 vincitore  su ~20 piloti, contro 3 su 20 per il podio); scale_pos_weight
    # calcolato dinamicamente si adatta automaticamente a entrambi i casi
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
    model.fit(X_train, y_train)

    return model


def find_best_threshold(model, test_df, thresholds=None, target_col="podium"):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    X_test = test_df[FEATURE_COL]
    y_test = test_df[target_col]
    y_proba = model.predict_proba(X_test)[:, 1]

    best_f1, best = -1.0, None
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best = {
                "threshold": t,
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1,
            }

    if best is None:
        raise ValueError("Nessuna soglia valida trovata")
    return best


def evaluate_model(model, test_df, threshold=0.6, target_col="podium"):
    X_test = test_df[FEATURE_COL]
    y_test = test_df[target_col]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    label_name = "Vincitore" if target_col == "winner" else "Podio"
    return classification_report(y_test, y_pred, target_names=[f"Non {label_name.lower()}", label_name])


def save_model(model, threshold, model_type="XGBoost", output_dir=None, name="model_final"):
    """
    Salva il modello e la sua configurazione. 'name' distingue i due modelli paralleli del progetto: 'model_final' (podio, default) e
    'model_winner' (vittoria), file separati, così una previsione può caricare entrambi senza conflitti.
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    model_path = f"{output_dir}/{name}.pkl"
    # Manteniamo il nome storico "model_config.json" per il modello podio, nome nuovo solo per il modello vincitore
    config_path = f"{output_dir}/model_config.json" if name == "model_final" else f"{output_dir}/{name}_config.json"

    joblib.dump(model, model_path)
    with open(config_path, "w") as f:
        json.dump({"threshold": threshold, "model_file": f"{name}.pkl", "model_type": model_type}, f, indent=2)

    print(f"Modello salvato in {model_path}")
    print(f"Configurazione salvata in {config_path}")