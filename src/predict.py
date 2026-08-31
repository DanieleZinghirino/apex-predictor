"""
Caricamento del modello salvato e generazione di nuove previsioni
"""
import joblib
import json
from src.train import FEATURE_COL
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_trained_model(model_dir=None):
    """
    Carica il modello addestrato e la sua configurazione
    
    Parametri:
        model_dir: cartella contenente il modello salvato. Se None, usa models/ relativo alla root del progetto
    
    Ritorna:
        tupla contenente (model, threshold)"""
    if model_dir is None:
        model_dir = os.path.join(PROJECT_ROOT, "models")
        
    with open(f"{model_dir}/model_config.json", "r") as f:
        config = json.load(f)

    model = joblib.load(f"{model_dir}/{config['model_file']}")
    threshold = config["threshold"]

    return model, threshold

def predict_podium(model, threshold, df):
    """
    Genera previsioni di podio per nuovi dati
    
    Parametri:
        model: modello addestrato
        threshold: soglia di decisione
        df: DataFrame con le stesse FEATURE_COLS usate in training
        
    Ritorna:
        Il DataFrame originale con due colonne aggiunte: 'podium_probability' e 'podium_predicted'
    """
    X = df[FEATURE_COL]
    proba = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["podium_probability"] = proba
    df["podium_predicted"] = (proba >= threshold).astype(int)

    return df