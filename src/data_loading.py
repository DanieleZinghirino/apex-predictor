"""
Funzioni per caricare e unire i dati grezzi
"""
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NA_VALUES = ["\\N"]

def load_raw_data(data_dir=None):
    """
    Carica le tabelle grezze necessarie al progetto
    
    Parametri:
        data_dir: percorso della cartella con i dati CSV grezzi. Se None, usa data/raw relativo alla root del progetto indipendentemente da dove viene lanciato lo script
        
    Ritorna:
        dizionario con chiavi 'races', 'results', 'drivers'. 'constructors', 'circuits' e i rispettivi DataFrame come valori
    """
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    
    races = pd.read_csv(f"{data_dir}/races.csv", na_values=NA_VALUES)
    results = pd.read_csv(f"{data_dir}/results.csv", na_values=NA_VALUES)
    drivers = pd.read_csv(f"{data_dir}/drivers.csv", na_values=NA_VALUES)
    constructors = pd.read_csv(f"{data_dir}/constructors.csv", na_values=NA_VALUES)
    circuits = pd.read_csv(f"{data_dir}/circuits.csv", na_values=NA_VALUES)

    return {
        "races": races,
        "results": results,
        "drivers": drivers,
        "constructors": constructors,
        "circuits": circuits,
    }

def build_working_dataset(races, results, min_year=2004):
    """
    Filtra le gare dal min_year in poi, unisce con i risultati, crea il target 'podium' e ordina cronologicamente
    
    Parametri:
        races: DataFrame delle gare
        results: DataFrames dei risultati
        min_year: anno minimo da includere (default 2004, decisione documentata in notebooks/01_eda.ipynb)
        
    Ritorna:
        DataFrame ordinato per data e driverId, con colonna 'podium'
    """
    races_recent = races[races["year"] >= min_year].copy()

    df = results.merge(
        races_recent[["raceId", "year", "round", "date", "circuitId"]],
        on="raceId",
        how="inner"
    )

    df["podium"] = df["positionOrder"] <= 3
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "driverId"]).reset_index(drop=True)

    return df





