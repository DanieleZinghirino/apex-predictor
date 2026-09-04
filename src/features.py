"""
Funzioni di feature engineering.

Ogni funzione calcola una feature usando solo dati precedenti alla riga corrente, per evitare data leakage temporale.
Il dataframe in input deve essere già ordinato per data
"""
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mappa nazionalità pilota/scuderia (es. "British") -> paese circuito (es. "UK")
# Le due fonti usano nomenclature diverse, serve un dizionario di conversione
NATIONALITY_TO_COUNTRY = {
    "British": "UK", "German": "Germany", "Spanish": "Spain",
    "Finnish": "Finland", "Dutch": "Netherlands", "French": "France",
    "Italian": "Italy", "Brazilian": "Brazil", "Australian": "Australia",
    "Austrian": "Austria", "Belgian": "Belgium", "Canadian": "Canada",
    "Danish": "Denmark", "Japanese": "Japan", "Mexican": "Mexico",
    "Monegasque": "Monaco", "American": "USA", "Thai": "Thailand",
    "Chinese": "China", "New Zealander": "New Zealand",
    # aggiungi altre nazionalità man mano che compaiono negli errori di merge
}

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
    # Standing: NaN alla primissima gara di una stagione: fallback su ultima posizione plausibile
    df["driver_standing_position"] = df["driver_standing_position"].fillna(df["driver_standing_position"].max())
    df["constructor_standing_position"] = df["constructor_standing_position"].fillna(df["constructor_standing_position"].max())

    # Teammate gap: NaN se non c'è compagno nella gara, fallback neutro (0)
    df["teammate_position_gap"] = df["teammate_position_gap"].fillna(0)

    # Qualifying gap: NaN se il pilota non ha tempo
    df["qualifying_gap_seconds"] = df["qualifying_gap_seconds"].fillna(df["qualifying_gap_seconds"].max())

    # Meteo: NaN se il circuito/data non ha dato disponibile, fallback su condizioni medie storiche
    df["race_max_temp_c"] = df["race_max_temp_c"].fillna(df["race_max_temp_c"].mean())
    df["race_precipitation_mm"] = df["race_precipitation_mm"].fillna(0)
    df["race_is_wet"] = df["race_is_wet"].fillna(0)

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


def add_home_race_flags(df, drivers_df, constructors_df, circuits_df):
    """
    Aggiunge due flag binari: il pilota corre nel proprio paese ('driver_home_race'), la scuderia ha sede nel paese della gara ('constructor_home_race')

    Parametri:
        df: DataFrame con driverId, constructorId, circuitId
        drivers_df, constructors_df, circuits_df: tabelle anagrafiche

    Ritorna:
        Il DataFrame con le due nuove colonne
    """
    df = df.copy()

    driver_country = drivers_df.set_index("driverId")["nationality"].map(NATIONALITY_TO_COUNTRY)
    constructor_country = constructors_df.set_index("constructorId")["nationality"].map(NATIONALITY_TO_COUNTRY)
    circuit_country = circuits_df.set_index("circuitId")["country"]

    df["_driver_country"] = df["driverId"].map(driver_country)
    df["_constructor_country"] = df["constructorId"].map(constructor_country)
    df["_circuit_country"] = df["circuitId"].map(circuit_country)

    df["driver_home_race"] = (df["_driver_country"] == df["_circuit_country"]).astype(int)
    df["constructor_home_race"] = (df["_constructor_country"] == df["_circuit_country"]).astype(int)

    df = df.drop(columns=["_driver_country", "_constructor_country", "_circuit_country"])
    return df


def add_standings_position(df, driver_standings_df, constructor_standings_df):
    """
    Aggiunge la posizione in classifica generale al momento di ciascuna gara. 
    Attenzione al leakage: driver_standings.csv registra la classifica DOPO ogni gara (raceId), quindi per la gara corrente dobbiamo usare la classifica della 
    gara precedente, non quella aggiornata dalla gara stessa (che includerebbe il suo stesso risultato).

    Parametri:
        df: DataFrame ordinato cronologicamente, con raceId, driverId, constructorId
        driver_standings_df, constructor_standings_df: tabelle standings

    Ritorna:
        Il DataFrame con 'driver_standing_position', 'constructor_standing_position'
    """
    df = df.copy()

    # Costruiamo, per ciascuna riga, il raceId della gara precedente dello stesso anno per lo stesso pilota/costruttore, 
    # poi cerchiamo la standing associata a quel raceId precedente
    race_order = df[["raceId", "year", "round"]].drop_duplicates().sort_values(["year", "round"])
    race_order["prev_raceId"] = race_order.groupby("year")["raceId"].shift(1)

    df = df.merge(race_order[["raceId", "prev_raceId"]], on="raceId", how="left")

    driver_pos = driver_standings_df.set_index(["raceId", "driverId"])["position"]
    df["driver_standing_position"] = df.apply(
        lambda r: driver_pos.get((r["prev_raceId"], r["driverId"])), axis=1
    )

    constructor_pos = constructor_standings_df.set_index(["raceId", "constructorId"])["position"]
    df["constructor_standing_position"] = df.apply(
        lambda r: constructor_pos.get((r["prev_raceId"], r["constructorId"])), axis=1
    )

    df = df.drop(columns=["prev_raceId"])
    return df


def add_teammate_comparison(df):
    """
    Confronta la forma recente del pilota con quella del suo compagno di squadra attuale

    Parametri:
        df: DataFrame con 'driver_recent_position_avg' già calcolata, più driverId, constructorId, raceId

    Ritorna:
        Il DataFrame con 'teammate_position_gap': positivo se il pilota va meglio del compagno, negativo se va peggio
    """
    df = df.copy()

    # Per ogni gara+scuderia, calcoliamo la media di driver_recent_position_avg tra i piloti di quella scuderia in quella gara (di norma 2)
    team_avg = df.groupby(["raceId", "constructorId"])["driver_recent_position_avg"].transform("mean")
    team_count = df.groupby(["raceId", "constructorId"])["driverId"].transform("count")

    # Il compagno ha come media: (somma_team - valore_proprio) / (count - 1)
    # Con 2 piloti per scuderia (caso standard), semplifica a: 2*media - valore_proprio
    teammate_avg = (team_avg * team_count - df["driver_recent_position_avg"]) / (team_count - 1)

    # Gap positivo = il pilota ha una posizione media migliore (numero più basso) del compagno, quindi teammate_avg - propria posizione > 0
    df["teammate_position_gap"] = teammate_avg - df["driver_recent_position_avg"]

    return df


def _qualifying_time_to_seconds(time_str):
    """
    Converte un tempo di qualifica dal formato 'M:SS.mmm' (stringa)
    a secondi (float). Ritorna None se il formato non è valido o
    il valore è mancante (pilota non ha segnato un tempo, es. eliminato
    prima di quella sessione).
    """
    if pd.isnull(time_str):
        return None
    try:
        minutes, rest = time_str.split(":")
        return int(minutes) * 60 + float(rest)
    except (ValueError, AttributeError):
        return None


def add_qualifying_gap(df, qualifying_df):
    """
    Aggiunge il distacco in qualifica dal miglior tempo della sessione in secondi. Usa il miglior tempo disponibile tra Q1/Q2/Q3 per ciascun pilota

    Parametri:
        df: DataFrame con raceId, driverId
        qualifying_df: DataFrame qualifying.csv (colonne q1, q2, q3)

    Ritorna:
        Il DataFrame con 'qualifying_gap_seconds'
    """
    df = df.copy()
    q = qualifying_df.copy()

    for col in ["q1", "q2", "q3"]:
        q[f"{col}_sec"] = q[col].apply(_qualifying_time_to_seconds)

    # Il miglior tempo personale è il minimo tra le sessioni disputate
    q["best_time_sec"] = q[["q1_sec", "q2_sec", "q3_sec"]].min(axis=1)

    # Il tempo di riferimento della gara è il minimo assoluto tra tutti i piloti
    pole_time = q.groupby("raceId")["best_time_sec"].transform("min")
    q["qualifying_gap_seconds"] = q["best_time_sec"] - pole_time

    df = df.merge(
        q[["raceId", "driverId", "qualifying_gap_seconds"]],
        on=["raceId", "driverId"], how="left"
    )

    return df


def add_weather_features(df, weather_path=None):
    """
    Unisce i dati meteo da data/reference/race_weather.csv, precalcolati con scripts/fetch_weather.py

    Parametri:
        df: DataFrame con raceId
        weather_path: percorso del CSV meteo; None usa il default

    Ritorna:
        Il DataFrame con 'race_max_temp_c', 'race_precipitation_mm', 'race_is_wet' (flag: precipitazioni > 1mm)
    """
    if weather_path is None:
        weather_path = os.path.join(PROJECT_ROOT, "data", "reference", "race_weather.csv")

    weather = pd.read_csv(weather_path)
    df = df.merge(weather, on="raceId", how="left")

    df = df.rename(columns={
        "max_temp_c": "race_max_temp_c",
        "precipitation_mm": "race_precipitation_mm",
    })
    df["race_is_wet"] = (df["race_precipitation_mm"] > 1.0).astype(int)

    return df


def build_all_features(df, circuits_df, drivers_df, constructors_df, driver_standings_df, constructor_standings_df, qualifying_df, n_races=10):
    """
    Applica in sequenza tutte le funzioni di feature engineering e la gestione dei valori mancanti. Punto di ingresso unico per costruire il dataset di feature completo

    Parametri:
        df: DataFrame base
        circuits_df: DataFrame circuits.csv, necessario per la mappatura circuitRef->circuitId delle caratteristiche statiche
        constructors_df: DataFrame constructors.csv, necessario per la mappatura nazionalità->paese
        drivers_df: DataFrame drivers.csv, necessario per la mappatura nazionalità->paese
        driver_standings_df, constructor_standings_df: DataFrame standings, necessari per calcolare la posizione in classifica prima di ciascuna gara
        qualifying_df: DataFrame qualifying.csv, necessario per calcolare il distacco in qualifica
        n_races: finestra per forma pilota e affidabilità scuderia

    Ritorna:
        DataFrame pronto per il training, con tutte le feature e senza NaN residui
    """
    df = add_driver_recent_form(df, n_races=n_races)
    df = add_constructor_reliability(df, n_races=n_races)
    df = add_driver_circuit_history(df)
    df = add_circuit_speed_and_overtaking(df)
    df = add_circuit_static_characteristics(df, circuits_df)
    df = add_home_race_flags(df, drivers_df, constructors_df, circuits_df)
    df = add_standings_position(df, driver_standings_df, constructor_standings_df)
    df = add_teammate_comparison(df)  # dopo add_driver_recent_form, ne dipende
    df = add_qualifying_gap(df, qualifying_df)
    df = add_weather_features(df)
    df = handle_missing_value(df)

    return df