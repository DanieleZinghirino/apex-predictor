"""
Training e valutazione del modello
"""
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_COL = [
    "grid",
    "driver_recent_points_avg",
    "driver_recent_position_avg",
    "constructor_reliability",
    "driver_circuit_avg_position",
    "no_circuit_history",
]
def temporal_split(df, test_start_year=2023):
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
    Allena il modello Random Forest definitivo (vedi confronto con Logistic Regression documentato nel README e nel notebook 02)

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
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    return model

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

def save_model(model, threshold=0.6, output_dir=None):
    """
    Salva il modello e la sua configurazione
    
    Parametri:
        model: modello addestrato da salvare
        threshold: soglia di decisione da salvare insieme al modello
        output_dir: cartella di destinazione. Se None, usa models/ relativo alla root del progetto
    """
    if output_dir is None:
        output_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(output_dir, exist_ok=True)

    model_path = f"{output_dir}/random_forest_final.pkl"
    config_path = f"{output_dir}/model_config.json"

    joblib.dump(model, model_path)
    with open(config_path, "w") as f:
        json.dump({"threshold": threshold, "model_file": "random_forest_final.pkl"}, f, indent=2)

    print(f"Modello salvato in {model_path}")
    print(f"Configurazione salvata in {config_path}")