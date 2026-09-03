"""
Funzioni di feature engineering.

Ogni funzione calcola una feature usando solo dati precedenti alla riga corrente, per evitare data leakage temporale.
Il dataframe in input deve essere già ordinato per data
"""
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_driver_recent_form(df, n_races=10):
    """
    Aggiunge la forma recente del pilota: media punti e posizione finale sulle ultime n_races gare precedenti
    
    Parametri:
        df: DataFrame ordinato cronologicamente, con colonne 'driverId', 'points', 'positionOrder'
        n_races: dimensione della finestra mobile (default 10)
        
    Ritorna:
        Il Dataframe con due nuove colonne: 'driver_recent_point_avg', 'driver_recent_position_avg'
    """
    df = df.copy()

    df["driver_recent_points_avg"] = (
        df.groupby("driverId")["points"]
        .apply(lambda x: x.shift(1).rolling(n_races, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    df["driver_recent_position_avg"] = (
            df.groupby("driverId")["positionOrder"]
            .apply(lambda x: x.shift(1).rolling(n_races, min_periods=1).mean())
            .reset_index(level=0, drop=True)
    )

    return df

def add_constructor_reliability(df, n_races=10):
    """
    Aggiunge l'affidabilità della scuderia: percentuale di gare completate sulle ultime n_races gare del costruttore
    
    Parametri:
        df: DataFrame ordinato cronologicamente, con colonne 'constructorId', 'position'
        n_races: dimensione della finestra mobile (default 10)
        
    Ritorna:
        Il DataFrame con la nuova colonna 'constructor reliability' e la colonna di appoggio 'finished'
    """
    df = df.copy()
    df["finished"] = df["position"].notnull().astype(int)

    df["constructor_reliability"] = (
        df.groupby("constructorId")["finished"]
        .apply(lambda x: x.shift(1).rolling(n_races, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )

    return df

def add_driver_circuit_history(df):
    """
    Aggiunge lo storico del pilota su ciascun circuito specifico: media poisizione finale su quel circuito su tutte le apparizioni precedenti.
    
    Parametri:
        df: DataFrame ordinato cronologicamente con colonne 'driverId', 'circuitId', 'positionOrder'
        
    Ritorna:
        Il DataFrame con la nuova colonna 'driver_circuit_avg_position'
    """
    df = df.copy()

    df["driver_circuit_avg_position"] = (
        df.groupby(["driverId", "circuitId"])["positionOrder"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=[0,1], drop=True)
    )

    return df

def handle_missing_value(df):
    """
    Applica la strategia di gestione dei NaN validata nell'EDA:
    - elimina le righe con NaN nelle feature strutturali (forma pilota, affidabilità scuderia), prime apparizioni in assoluto quindi nessun modo sensato di stimarle;
    - per lo storico del circuito, usa la forma generale del pilota come fallback, con un flag esplicito 'no_circuit_history'
    
    Parametri:
        df: DataFrame con tutte le feature già calcolate
    
    Ritorna:
        DataFrame pulito, senza NaN residui nelle feature del modello
    """
    df = df.dropna(subset=[
        "driver_recent_points_avg",
        "driver_recent_position_avg",
        "constructor_reliability"
    ]).copy()

    df["no_circuit_history"] = df["driver_circuit_avg_position"].isnull().astype(int)
    df["driver_circuit_avg_position"] = df["driver_circuit_avg_position"].fillna(df["driver_recent_position_avg"])

    # Prima apparizione di un circuito nello storico: nessun dato pregresso su velocità/sorpassi per quel circuito. 
    # Fallback sulla media globale di tutti i circuiti, meglio di eliminare la riga
    df["circuit_avg_speed_history"] = df["circuit_avg_speed_history"].fillna(
        df["circuit_avg_speed_history"].mean()
    )
    df["circuit_overtaking_index"] = df["circuit_overtaking_index"].fillna(
        df["circuit_overtaking_index"].mean()
    )

    return df


def add_circuit_static_characteristics(df, circuits_df, characteristics_path=None):
    """
    Unisce le caratteristiche fisiche statiche del circuito dati compilati manualmente in data/reference/circuit_characteristics.csv, 
    verificati incrociando più fonti (vedi README per la metodologia).

    Parametri:
        df: DataFrame con colonna 'circuitId'
        circuits_df: DataFrame circuits.csv
        characteristics_path: percorso del CSV; se None usa il default in data/reference/

    Ritorna:
        Il DataFrame con le nuove colonne: 'circuit_length_km', 'circuit_num_corners', 'circuit_altitude_m', 'circuit_downforce_medium',
        'circuit_downforce_high', 'circuit_data_missing'
        (flag: 1 se il circuito non è nella tabella di riferimento)
    """
    if characteristics_path is None:
        characteristics_path = os.path.join(PROJECT_ROOT, "data", "reference", "circuit_characteristics.csv")

    characteristics = pd.read_csv(characteristics_path)

    # Ponte circuitRef (testuale, nella tabella) -> circuitId (numerico, nel dataset)
    ref_to_id = dict(zip(circuits_df["circuitRef"], circuits_df["circuitId"]))
    characteristics["circuitId"] = characteristics["circuit_ref"].map(ref_to_id)

    # One-hot encoding manuale per downforce_level
    characteristics["circuit_downforce_medium"] = (characteristics["downforce_level"] == "medium").astype(int)
    characteristics["circuit_downforce_high"] = (characteristics["downforce_level"] == "high").astype(int)

    merge_cols = ["circuitId", "length_km", "num_corners", "altitude_m",
                  "circuit_downforce_medium", "circuit_downforce_high"]
    df = df.merge(
        characteristics[merge_cols].rename(columns={
            "length_km": "circuit_length_km",
            "num_corners": "circuit_num_corners",
            "altitude_m": "circuit_altitude_m",
        }),
        on="circuitId", how="left"
    )

    # Flag esplicito per circuiti mancanti dalla tabella invece di lasciare NaN silenziosi
    df["circuit_data_missing"] = df["circuit_length_km"].isnull().astype(int)

    return df


def add_circuit_speed_and_overtaking(df):
    """
    Calcola, per ciascun circuito, la velocità media storica e un indice di quanto cambiano le posizioni tra griglia e arrivo, usando gare precedenti a quella corrente

    Parametri:
        df: DataFrame ordinato cronologicamente, con colonne 'circuitId', 'fastestLapSpeed', 'grid', 'positionOrder'

    Ritorna:
        Il DataFrame con due nuove colonne: 'circuit_avg_speed_history', 'circuit_overtaking_index'
    """
    df = df.copy()

    # overtaking_index: |grid - positionOrder| per riga, poi mediato per circuito su tutte le apparizioni precedenti 
    # (expanding, non rolling: come per driver_circuit_avg_position, un circuito ricorre solo ~1 volta a stagione, poca storia per finestra fissa)
    df["_position_change"] = (df["grid"] - df["positionOrder"]).abs()

    df["circuit_overtaking_index"] = (
        df.groupby("circuitId")["_position_change"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df["circuit_avg_speed_history"] = (
        df.groupby("circuitId")["fastestLapSpeed"]
        .apply(lambda x: x.shift(1).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    df = df.drop(columns=["_position_change"])
    return df

def build_all_features(df, circuits_df, n_races=10):
    """
    Applica in sequenza tutte le funzioni di feature engineering e la gestione dei valori mancanti. Punto di ingresso unico per costruire il dataset di feature completo

    Parametri:
        df: DataFrame base
        circuits_df: DataFrame circuits.csv, necessario per la mappatura circuitRef->circuitId delle caratteristiche statiche
        n_races: finestra per forma pilota e affidabilità scuderia

    Ritorna:
        DataFrame pronto per il training, con tutte le feature e senza NaN residui
    """
    df = add_driver_recent_form(df, n_races=n_races)
    df = add_constructor_reliability(df, n_races=n_races)
    df = add_driver_circuit_history(df)
    df = add_circuit_speed_and_overtaking(df)
    df = add_circuit_static_characteristics(df, circuits_df)
    df = handle_missing_value(df)

    return df