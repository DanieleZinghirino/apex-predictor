"""
Costruzione feature e generazione previsioni per una gara futura usando lo storico locale aggiornato e la griglia di partenza live da Jolpica-F1
"""
import pandas as pd

N_RACES_FORM = 10
N_RACES_RELIABILITY = 10


def map_refs_to_ids(qualifying_results, drivers_df, constructors_df):
    """
    Converte driver_ref o constructor_ref (Jolpica) in driverId o constructorId
    
    Parametri:
        qualifying_results: lista di dizionari da get_qualifying_results()
        drivers_df, constructors_df: DataFrame con le colonne driverRef o constructorRef (da load_raw_data())
    
    Ritorna:
        DataFrame con driverId, constructorId, grid e i ref originali
    """
    driver_map = dict(zip(drivers_df["driverRef"], drivers_df["driverId"]))
    constructor_map = dict(zip(constructors_df["constructorRef"], constructors_df["constructorId"]))

    rows = []
    for r in qualifying_results:
        rows.append({
            "driver_ref": r["driver_ref"],
            "constructor_ref": r["constructor_ref"],
            "driverId": driver_map.get(r["driver_ref"]),  # None se non mappabile
            "constructorId": constructor_map.get(r["constructor_ref"]),
            "grid": r["grid_position"],
            "qualifying_gap_seconds": r.get("qualifying_gap_seconds", 0.0),
        })

    df = pd.DataFrame(rows)

    unmapped = df[df["driverId"].isnull()]
    if not unmapped.empty:
        print(f"ATTENZIONE: piloti non mappati (esclusi dalla previsione): "
              f"{unmapped['driver_ref'].tolist()}")

    return df.dropna(subset=["driverId", "constructorId"])


def compute_circuit_features(circuit_id, historical_df, circuits_df, characteristics_path=None):
    """
    Calcola le feature del circuito per una gara futura, con la stessa logica di src/features.py
    """
    import os
    import pandas as pd
    from src.data_loading import PROJECT_ROOT

    if characteristics_path is None:
        characteristics_path = os.path.join(PROJECT_ROOT, "data", "reference", "circuit_characteristics.csv")

    characteristics = pd.read_csv(characteristics_path)
    ref_to_id = dict(zip(circuits_df["circuitRef"], circuits_df["circuitId"]))
    characteristics["circuitId"] = characteristics["circuit_ref"].map(ref_to_id)

    static_row = characteristics[characteristics["circuitId"] == circuit_id]

    if static_row.empty:
        static = {
            "circuit_length_km": characteristics["length_km"].mean(),
            "circuit_num_corners": characteristics["num_corners"].mean(),
            "circuit_altitude_m": characteristics["altitude_m"].mean(),
            "circuit_downforce_medium": 0,
            "circuit_downforce_high": 0,
        }
    else:
        row = static_row.iloc[0]
        static = {
            "circuit_length_km": row["length_km"],
            "circuit_num_corners": row["num_corners"],
            "circuit_altitude_m": row["altitude_m"],
            "circuit_downforce_medium": int(row["downforce_level"] == "medium"),
            "circuit_downforce_high": int(row["downforce_level"] == "high"),
        }

    circuit_races = historical_df[historical_df["circuitId"] == circuit_id]

    if circuit_races.empty:
        avg_speed = historical_df["fastestLapSpeed"].mean()
        overtaking = (historical_df["grid"] - historical_df["positionOrder"]).abs().mean()
    else:
        avg_speed = circuit_races["fastestLapSpeed"].mean()
        overtaking = (circuit_races["grid"] - circuit_races["positionOrder"]).abs().mean()

    static["circuit_avg_speed_history"] = avg_speed
    static["circuit_overtaking_index"] = overtaking

    return static


def compute_constructor_reliability(constructor_id, historical_df, n_races=N_RACES_RELIABILITY):
    constructor_races = historical_df[historical_df["constructorId"] == constructor_id].sort_values("date")

    if constructor_races.empty:
        return None

    recent = constructor_races.tail(n_races)
    finished = recent["position"].notnull().astype(int)
    return finished.mean()


def compute_circuit_history(driver_id, circuit_id, historical_df):
    races_here = historical_df[
        (historical_df["driverId"] == driver_id) &
        (historical_df["circuitId"] == circuit_id)
    ]

    if races_here.empty:
        return None, 1

    return races_here["positionOrder"].mean(), 0


def compute_driver_form(driver_id, historical_df, n_races=N_RACES_FORM):
    """
    Media mobile esponenziale di punti e posizione finale del pilota su tutto lo storico disponibile
    """
    driver_races = historical_df[historical_df["driverId"] == driver_id].sort_values("date")

    if driver_races.empty:
        return None, None

    points_avg = driver_races["points"].ewm(span=n_races, min_periods=1).mean().iloc[-1]
    position_avg = driver_races["positionOrder"].ewm(span=n_races, min_periods=1).mean().iloc[-1]

    return points_avg, position_avg


def compute_driver_recent_grid_avg(driver_id, historical_df, n_races=6):
    """
    Stima la posizione di griglia per una gara futura usando la mediana delle ultime n_races griglie

    Parametri:
        driver_id: ID interno del pilota
        historical_df: DataFrame storico
        n_races: quante gare recenti considerare

    Ritorna:
        float, oppure None se nessuno storico disponibile
    """
    driver_races = historical_df[historical_df["driverId"] == driver_id].sort_values("date")

    if driver_races.empty:
        return None

    return driver_races["grid"].tail(n_races).median()

def get_current_roster(historical_df, season):
    season_races = historical_df[historical_df["year"] == season]

    if season_races.empty:
        # Fallback all'ultima stagione disponibile nello storico
        latest_season = historical_df["year"].max()
        season_races = historical_df[historical_df["year"] == latest_season]

    last_race_id = season_races["raceId"].max()
    roster = season_races[season_races["raceId"] == last_race_id][["driverId", "constructorId"]]

    return roster.drop_duplicates()


def _get_latest_standings(driver_id, constructor_id, historical_df):
    """Estrae la posizione in classifica più recente per pilota e costruttore."""
    driver_races = historical_df[historical_df["driverId"] == driver_id]
    constructor_races = historical_df[historical_df["constructorId"] == constructor_id]

    driver_pos = driver_races["driver_standing_position"].dropna().iloc[-1] if not driver_races.empty and "driver_standing_position" in driver_races.columns else 10
    constructor_pos = constructor_races["constructor_standing_position"].dropna().iloc[-1] if not constructor_races.empty and "constructor_standing_position" in constructor_races.columns else 5

    return driver_pos, constructor_pos


def _enrich_missing_feature_columns(df):
    """
    Garantisce che tutte le feature richieste dal modello (FEATURE_COL) siano presenti con valori standard/fallback dove necessario
    """
    df = df.copy()
    df["teammate_position_gap"] = df.groupby("constructorId")["grid"].transform(lambda x: x - x.mean()).fillna(0)

    # Feature di qualifica e compagno di squadra
    if "qualifying_gap_seconds" not in df.columns:
        df["qualifying_gap_seconds"] = 0.0

    if "teammate_position_gap" not in df.columns:
        team_avg = df.groupby("constructorId")["driver_recent_position_avg"].transform("mean")
        team_count = df.groupby("constructorId")["driverId"].transform("count")
        denom = (team_count - 1).replace(0, 1)  # evita divisione per zero se un pilota è solo in scuderia
        teammate_avg = (team_avg * team_count - df["driver_recent_position_avg"]) / denom
        df["teammate_position_gap"] = (teammate_avg - df["driver_recent_position_avg"]).fillna(0)

    # Feature gara di casa (valori predefiniti a 0 se non calcolati esplicitamente)
    if "driver_home_race" not in df.columns:
        df["driver_home_race"] = 0
    if "constructor_home_race" not in df.columns:
        df["constructor_home_race"] = 0

    # Feature meteo (valori predefiniti per gare standard)
    df["race_max_temp_c"] = df["race_max_temp_c"].fillna(25.0) if "race_max_temp_c" in df.columns else 25.0
    df["race_precipitation_mm"] = df["race_precipitation_mm"].fillna(0.0) if "race_precipitation_mm" in df.columns else 0.0
    df["race_is_wet"] = df["race_is_wet"].fillna(0) if "race_is_wet" in df.columns else 0

    return df


def build_upcoming_race_features(qualifying_df, historical_df, circuit_id, circuits_df, race_date=None):
    """
    Costruisce il DataFrame di feature per una gara futura dopo le qualifiche
    """
    circuit_features = compute_circuit_features(circuit_id, historical_df, circuits_df)
    weather_features = compute_live_weather(circuit_id, circuits_df, race_date) if race_date else {}
    rows = []

    for _, r in qualifying_df.iterrows():
        points_avg, position_avg = compute_driver_form(r["driverId"], historical_df)
        reliability = compute_constructor_reliability(r["constructorId"], historical_df)
        circuit_avg, no_history = compute_circuit_history(r["driverId"], circuit_id, historical_df)
        driver_pos, constructor_pos = _get_latest_standings(r["driverId"], r["constructorId"], historical_df)

        if points_avg is None:
            print(f"  ATTENZIONE: nessuno storico per driverId={r['driverId']}, escluso dalla previsione")
            continue

        if circuit_avg is None:
            circuit_avg = position_avg

        row = {
            "driverId": r["driverId"],
            "constructorId": r["constructorId"],
            "grid": r["grid"],
            "qualifying_gap_seconds": r.get("qualifying_gap_seconds", 0.0),
            "driver_recent_points_avg": points_avg,
            "driver_recent_position_avg": position_avg,
            "constructor_reliability": reliability if reliability is not None else 1.0,
            "driver_circuit_avg_position": circuit_avg,
            "no_circuit_history": no_history,
            "driver_standing_position": driver_pos,
            "constructor_standing_position": constructor_pos,
        }
        row.update(circuit_features)
        row.update(weather_features)
        rows.append(row)


    df = pd.DataFrame(rows)
    return _enrich_missing_feature_columns(df)


def build_pre_qualifying_features(historical_df, circuit_id, season, circuits_df, race_date=None):
    """
    Costruisce feature per una previsione anticipata prima delle qualifiche

    Parametri:
        historical_df: DataFrame storico
        circuit_id: ID interno del circuito della prossima gara
        season: anno della stagione corrente
        circuits_df: DataFrame circuits.csv
        race_date: data della gara, per il meteo (opzionale)

    Ritorna:
        DataFrame con le FEATURE_COL, più 'is_estimated_grid' per
        marcare esplicitamente la stima
    """
    circuit_features = compute_circuit_features(circuit_id, historical_df, circuits_df)
    weather_features = compute_live_weather(circuit_id, circuits_df, race_date) if race_date else {}

    roster = get_current_roster(historical_df, season)

    # Primo passaggio: calcoliamo tutti i valori grezzi per ciascun
    # pilota, SENZA ancora assegnare la griglia finale — ci serve
    # prima l'intero gruppo per poterli ordinare tra loro
    raw_rows = []
    for _, r in roster.iterrows():
        estimated_grid_raw = compute_driver_recent_grid_avg(r["driverId"], historical_df)
        points_avg, position_avg = compute_driver_form(r["driverId"], historical_df)
        reliability = compute_constructor_reliability(r["constructorId"], historical_df)
        circuit_avg, no_history = compute_circuit_history(r["driverId"], circuit_id, historical_df)
        driver_pos, constructor_pos = _get_latest_standings(r["driverId"], r["constructorId"], historical_df)

        if points_avg is None or estimated_grid_raw is None:
            continue

        if circuit_avg is None:
            circuit_avg = position_avg

        raw_rows.append({
            "driverId": r["driverId"],
            "constructorId": r["constructorId"],
            "estimated_grid_raw": estimated_grid_raw,  # solo per ordinare, non finisce nel modello
            "qualifying_gap_seconds": 0.0,
            "driver_recent_points_avg": points_avg,
            "driver_recent_position_avg": position_avg,  # riusata anche come spareggio
            "constructor_reliability": reliability if reliability is not None else 1.0,
            "driver_circuit_avg_position": circuit_avg,
            "no_circuit_history": no_history,
            "driver_standing_position": driver_pos,
            "constructor_standing_position": constructor_pos,
        })

    # Ordiniamo per stima di griglia crescente; a parità di stima, il pilota con forma recente migliore (position_avg più basso) va davanti
    raw_rows.sort(key=lambda x: (x["estimated_grid_raw"], x["driver_recent_position_avg"]))

    # Secondo passaggio: assegniamo la griglia finale come semplice posizione nella lista ordinata
    rows = []
    for position, raw_row in enumerate(raw_rows, start=1):
        row = {
            "driverId": raw_row["driverId"],
            "constructorId": raw_row["constructorId"],
            "grid": position,  # <-- intero simulato, non più la stima grezza
            "qualifying_gap_seconds": raw_row["qualifying_gap_seconds"],
            "driver_recent_points_avg": raw_row["driver_recent_points_avg"],
            "driver_recent_position_avg": raw_row["driver_recent_position_avg"],
            "constructor_reliability": raw_row["constructor_reliability"],
            "driver_circuit_avg_position": raw_row["driver_circuit_avg_position"],
            "no_circuit_history": raw_row["no_circuit_history"],
            "driver_standing_position": raw_row["driver_standing_position"],
            "constructor_standing_position": raw_row["constructor_standing_position"],
            "is_estimated_grid": 1,
        }
        row.update(circuit_features)
        row.update(weather_features)
        rows.append(row)

    df = pd.DataFrame(rows)
    return _enrich_missing_feature_columns(df)


def compute_live_qualifying_gaps(qualifying_results):
    """
    Calcola il distacco dal poleman in secondi, dai tempi Q1/Q2/Q3 grezzi restituiti da Jolpica

    Parametri:
        qualifying_results: lista di dict da get_qualifying_results()

    Ritorna:
        La stessa lista, con 'qualifying_gap_seconds' aggiunta a ciascun dict
    """
    from src.features import _qualifying_time_to_seconds

    for r in qualifying_results:
        times = [_qualifying_time_to_seconds(r.get(k)) for k in ("q1", "q2", "q3")]
        times = [t for t in times if t is not None]
        r["best_time_sec"] = min(times) if times else None

    valid_times = [r["best_time_sec"] for r in qualifying_results if r["best_time_sec"] is not None]
    pole_time = min(valid_times) if valid_times else None

    for r in qualifying_results:
        if r["best_time_sec"] is not None and pole_time is not None:
            r["qualifying_gap_seconds"] = r["best_time_sec"] - pole_time
        else:
            r["qualifying_gap_seconds"] = None

    return qualifying_results

def compute_live_weather(circuit_id, circuits_df, race_date):
    """
    Recupera la previsione meteo reale per la gara, usando lat/lng del circuito e la data della gara. Se il fetch fallisce (es. data
    troppo lontana per l'orizzonte di previsione), ritorna valori None, gestiti a valle come fallback esplicito.

    Parametri:
        circuit_id: ID interno del circuito
        circuits_df: DataFrame circuits.csv (colonne lat, lng)
        race_date: data della gara, formato 'YYYY-MM-DD'

    Ritorna:
        dict con race_max_temp_c, race_precipitation_mm, race_is_wet
    """
    from src.weather_client import get_weather_forecast

    circuit_row = circuits_df[circuits_df["circuitId"] == circuit_id]

    if circuit_row.empty:
        return {"race_max_temp_c": None, "race_precipitation_mm": None, "race_is_wet": None}

    lat, lng = circuit_row.iloc[0]["lat"], circuit_row.iloc[0]["lng"]
    forecast = get_weather_forecast(lat, lng, race_date)

    if forecast is None:
        return {"race_max_temp_c": None, "race_precipitation_mm": None, "race_is_wet": None}

    return {
        "race_max_temp_c": forecast["max_temp_c"],
        "race_precipitation_mm": forecast["precipitation_mm"],
        "race_is_wet": int(forecast["precipitation_mm"] > 1.0),
    }