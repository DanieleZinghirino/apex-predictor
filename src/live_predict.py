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
        drivers_df, constructors_df: DataFrame con le colonne driverRef o constructorRef (da liad_raw_data())
    
    Ritorna:
        DataFrame con driverId, constructorId, grid_posizion e i ref originali
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
        })

    df = pd.DataFrame(rows)

    unmapped = df[df["driverId"].isnull()]
    if not unmapped.empty:
        print(f"ATTENZIONE: piloti non mappati (esclusi dalla previsione): "
              f"{unmapped['driver_ref'].tolist()}")

    return df.dropna(subset=["driverId", "constructorId"])


def compute_driver_form(driver_id, historical_df, n_races=N_RACES_FORM):
    """
    Media punti e posizione finale del pilota nelle ultime n_races gare disputate

    Parametri:
        driver_id: ID interno del pilota
        historical_df: DataFrame storico (output di build_working_dataset)
        n_races: quante gare recenti considerare

    Ritorna:
        tupla (points_avg, position_avg). Se il pilota non ha storico ritorna (None, None)
    """
    driver_races = historical_df[historical_df["driverId"] == driver_id].sort_values("date")

    if driver_races.empty:
        return None, None

    recent = driver_races.tail(n_races)
    return recent["points"].mean(), recent["positionOrder"].mean()


def compute_constructor_reliability(constructor_id, historical_df, n_races=N_RACES_RELIABILITY):
    """
    Percentuale di gare completate dal costruttore nelle ultime n_races gare

    Ritorna:
        float tra 0 e 1, oppure None se nessuno storico disponibile
    """
    constructor_races = historical_df[historical_df["constructorId"] == constructor_id].sort_values("date")

    if constructor_races.empty:
        return None

    recent = constructor_races.tail(n_races)
    finished = recent["position"].notnull().astype(int)
    return finished.mean()

def compute_circuit_history(driver_id, circuit_id, historical_df):
    """
    Media posizione finale del pilota su quel circuito specifico, su tutte le apparizioni storiche disponibili.

    Ritorna:
        tupla (avg_position, no_history_flag). avg_position è None
        se il pilota non ha mai corso su quel circuito.
    """
    races_here = historical_df[
        (historical_df["driverId"] == driver_id) &
        (historical_df["circuitId"] == circuit_id)
    ]

    if races_here.empty:
        return None, 1

    return races_here["positionOrder"].mean(), 0


def build_upcoming_race_features(qualifying_df, historical_df, circuit_id):
    """
    Costruisce il DataFrame di feature per una gara futura, pronto per essere passato al modello

    Parametri:
        qualifying_df: output di map_refs_to_ids() — driverId,constructorId, grid già mappati
        historical_df: DataFrame storico
        circuit_id: ID interno del circuito della gara da prevedere

    Ritorna:
        DataFrame con tutte le FEATURE_COL pronte per src.predict.predict_podium,
        più driverId/constructorId per identificare ciascuna riga
    """
    rows = []

    for _, r in qualifying_df.iterrows():
        points_avg, position_avg = compute_driver_form(r["driverId"], historical_df)
        reliability = compute_constructor_reliability(r["constructorId"], historical_df)
        circuit_avg, no_history = compute_circuit_history(r["driverId"], circuit_id, historical_df)

        # Se manca la forma generale del pilota non possiamo nemmeno applicare il fallback usato in src/features.py, segnaliamo e saltiamo invece di inventare un valore
        if points_avg is None:
            print(f"  ATTENZIONE: nessuno storico per driverId={r['driverId']}, escluso dalla previsione")
            continue

        # Fallback identico a quello usato in training: se manca lo storico specifico sul circuito, usa la forma generale
        if circuit_avg is None:
            circuit_avg = position_avg

        rows.append({
            "driverId": r["driverId"],
            "constructorId": r["constructorId"],
            "grid": r["grid"],
            "driver_recent_points_avg": points_avg,
            "driver_recent_position_avg": position_avg,
            "constructor_reliability": reliability if reliability is not None else 1.0,
            "driver_circuit_avg_position": circuit_avg,
            "no_circuit_history": no_history,
        })

    return pd.DataFrame(rows)


def compute_driver_recent_grid_avg(driver_id, historical_df, n_races=5):
    """
    Media della posizione di griglia nelle ultime n_races gare del pilota, usata come STIMA della griglia quando le qualifiche reali non sono ancora disponibili. 
    Meno affidabile della griglia vera

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

    return driver_races.tail(n_races)["grid"].mean()


def get_current_roster(historical_df, season):
    """
    Ritorna la formazione piloti/costruttori dell'ultima gara disputata in una stagione

    Parametri:
        historical_df: DataFrame storico
        season: anno della stagione corrente

    Ritorna:
        DataFrame con driverId, constructorId univoci
    """
    season_races = historical_df[historical_df["year"] == season]

    if season_races.empty:
        return pd.DataFrame(columns=["driverId", "constructorId"])

    last_race_id = season_races["raceId"].max()
    roster = season_races[season_races["raceId"] == last_race_id][["driverId", "constructorId"]]

    return roster.drop_duplicates()


def build_pre_qualifying_features(historical_df, circuit_id, season):
    """
    Costruisce feature per una previsione anticipata, prima che le qualifiche reali siano disponibili 
    Usa la griglia stimata (media recente) invece di quella reale, e la formazione piloti dell'ultima gara disputata come proxy degli iscritti.

    Parametri:
        historical_df: DataFrame storico
        circuit_id: ID interno del circuito della prossima gara
        season: anno della stagione corrente

    Ritorna:
        DataFrame con le FEATURE_COLS, più una colonna
        'is_estimated_grid' per marcare esplicitamente la stima
    """
    roster = get_current_roster(historical_df, season)
    rows = []

    for _, r in roster.iterrows():
        estimated_grid = compute_driver_recent_grid_avg(r["driverId"], historical_df)
        points_avg, position_avg = compute_driver_form(r["driverId"], historical_df)
        reliability = compute_constructor_reliability(r["constructorId"], historical_df)
        circuit_avg, no_history = compute_circuit_history(r["driverId"], circuit_id, historical_df)

        if points_avg is None or estimated_grid is None:
            continue

        if circuit_avg is None:
            circuit_avg = position_avg

        rows.append({
            "driverId": r["driverId"],
            "constructorId": r["constructorId"],
            "grid": estimated_grid,
            "driver_recent_points_avg": points_avg,
            "driver_recent_position_avg": position_avg,
            "constructor_reliability": reliability if reliability is not None else 1.0,
            "driver_circuit_avg_position": circuit_avg,
            "no_circuit_history": no_history,
            "is_estimated_grid": 1,
        })

    return pd.DataFrame(rows)
