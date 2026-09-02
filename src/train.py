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

os.makedirs("../models", exist_ok=True)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_COL = [
    "grid",
    "driver_recent_points_avg",
    "driver_recent_position_avg",
    "constructor_reliability",
    "driver_circuit_avg_position",
    "no_circuit_history",
]
# Iperparametri del modello campione, trovati con RandomizedSearchCV +TimeSeriesSplit (vedi notebooks/04_hyperparameter_tuning.ipynb)
XGB_CHAMPION_PARAMS = {
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

def train_champion_model(train_df, params=None):
    """
    Allena il modello campione del progetto: XGBoost con gli iperparametri trovati dal tuning (vedi notebooks/04).

    Parametri:
        train_df: DataFrame di training, con FEATURE_COLS e 'podium'
        params: dict di iperparametri XGBoost; se None usa
                XGB_CHAMPION_PARAMS (la configurazione validata)

    Ritorna:
        Il modello XGBClassifier addestrato
    """
    if params is None:
        params = XGB_CHAMPION_PARAMS

    X_train = train_df[FEATURE_COL]
    y_train = train_df["podium"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, **params)
    model.fit(X_train, y_train)

    return model


def find_best_threshold(model, test_df, thresholds=None):
    """
    Cerca, tra un insieme di soglie candidate, quella che massimizza l'F1-score sulla classe podio.

    Parametri:
        model: modello addestrato con predict_proba
        test_df: DataFrame di test, con FEATURE_COLS e 'podium'
        thresholds: array di soglie da provare; se None usa
                    np.arange(0.1, 0.95, 0.05)

    Ritorna:
        dict con threshold, precision, recall, f1 al punto migliore

    Solleva:
        ValueError se 'thresholds' è vuoto (nessuna soglia da provare)
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    X_test = test_df[FEATURE_COL]
    y_test = test_df["podium"]
    y_proba = model.predict_proba(X_test)[:, 1]

    best_f1 = -1.0
    best = None
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
        raise ValueError("Nessuna soglia valida trovata: 'thresholds' è vuoto?")

    return best


def evaluate_model(model, test_df, threshold=0.6):
    """
    Valuta il modello sul test set, applicando la soglia di decisione ottimale ricavata dall'analisi nel notebook 02

    Parametri:
        model: modello addestrato
        test_df: DataFrame di test
        threshold: soglia di decisione

    Ritorna:
        Il classification_report come stringa
    """
    X_test = test_df[FEATURE_COL]
    y_test = test_df["podium"]

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred = (y_proba > threshold).astype(int)

    return classification_report(y_test, y_pred, target_names=["Non podio", "Podio"])


def save_model(model, threshold, model_type="XGBoost", output_dir=None):
    """
    Salva il modello e la sua configurazione

    Parametri:
        model: modello addestrato da salvare
        threshold: soglia di decisione da salvare insieme al modello
        model_type: etichetta leggibile del tipo di modello, salvata
                    nella configurazione per riferimento (non usata
                    dal codice, solo documentativa)
        output_dir: cartella di destinazione. Se None, usa models/
                    relativo alla root del progetto
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models")

    os.makedirs(output_dir, exist_ok=True)

    model_path = f"{output_dir}/model_final.pkl"
    config_path = f"{output_dir}/model_config.json"

    joblib.dump(model, model_path)
    with open(config_path, "w") as f:
        json.dump({
            "threshold": threshold,
            "model_file": "model_final.pkl",
            "model_type": model_type,
        }, f, indent=2)

    print(f"Modello salvato in {model_path}")
    print(f"Configurazione salvata in {config_path}")