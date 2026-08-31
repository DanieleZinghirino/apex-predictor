"""
Funzioni di feature engineering.

Ogni funzione calcola una feature usando solo dati precedenti alla riga corrente, per evitare data leakage temporale.
Il dataframe in input deve essere già ordinato per data
"""
import pandas as pd

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

    return df

def build_all_features(df, n_races=10):
    """
    Applica in sequenza tutte le funzioni di feature engineering e la gestione dei valori mancanti. Punto di ingresso unico per costruire il dataset di feature completo

    Parametri:
        df: DataFrame base
        n_races: finestra per forma pilota e affidabilità scuderia

    Ritorna:
        DataFrame pronto per il training, con tutte le feature e senza NaN residui
    """
    df = add_driver_recent_form(df, n_races=n_races)
    df = add_constructor_reliability(df, n_races=n_races)
    df = add_driver_circuit_history(df)
    df = handle_missing_value(df)

    return df
