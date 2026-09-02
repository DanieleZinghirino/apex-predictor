"""
Script di backfill: scarica i risultati delle gare dal 2025 da Jolpica-F1 e li integra nel dataset storico locale
Idempotente: eseguibile più volte senza duplicare i dati
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from datetime import date

from src.data_loading import load_raw_data, PROJECT_ROOT
from src.jolpica_client import get_season_races, get_race_results

SEASONS_TO_CHECK = [2025, 2026]

def build_ref_maps(data):
    """
    Costruisce dizionari di mappatura da riferimento testuale a ID numerico interno

    Ritorna:
        tupla (driver_map, constructor_map, circuit_map), ciascuna con un dizionario {ref_testuale: id_numerico}
    """
    driver_map = dict(zip(data["drivers"]["driverRef"], data["drivers"]["driverId"]))
    constructor_map = dict(zip(data["constructors"]["constructorRef"], data["constructors"]["constructorId"]))
    circuit_map = dict(zip(data["circuits"]["circuitRef"], data["circuits"]["circuitId"]))

    return driver_map, constructor_map, circuit_map


def get_existing_races(races_df):
    """
    Ritorna l'insieme delle combinazioni (year, round) già presenti nello storico locale
    
    Parametri:
        races_df: DataFrame 'races.csv'
    
    Ritorna:
        set di tuple (year, round)
    """
    return set(zip(races_df["year"], races_df["round"]))


def find_new_races(seasons, existing_races, cutoff_date=None):
    """
    Interroga Jolpica per ciascuna stagione e ritorna le gare che non sono ancora nello storico locale e la cui data è già passata
    
    Parametri:
        seasons: lista di anni da controllare
        existing_races: set di (year, round) già presenti in locale (ottenibili da get_existing_races)
        cutoff_date: date oltre la quale non considerare le gare (default: oggi)
        
    """
    if cutoff_date is None:
        cutoff_date = date.today()

    new_races = []

    for season in seasons:
        season_races = get_season_races(season)
        for race in season_races:
            race_date = pd.to_datetime(race["date"]).date()

            if(season, race["round"]) in existing_races:
                continue    # già nello storico locale

            if race_date > cutoff_date:
                continue    # gara futura

            new_races.append({**race, "season": season})

    return sorted(new_races, key=lambda r: (r["season"], r["round"]))


def resolve_or_create_id(ref, ref_map, id_column_name, entity_df, extra_fields=None):
    """
    Trova l'ID numerico corrispondente a un riferimento testuale (driverRef o constructorRef): se non esiste ancora, crea un nuovo ID e una nuova riga nel DataFrame 
    dell'entità, aggiornando anche la mappa
    
    Parametri:
        ref: il riferimento testuale
        ref_map: dizionario {ref: id} da aggiornare se serve un nuovo ID
        id_column_name: nome della colonna ID (driverId, constructorId, etc.)
        entity_df: il DataFrame a cui eventualmente aggiungere una nuova riga (drivers, constructors, etc.)
        extra_fields: dizionario di campi aggiuntivi per la nuova riga
    
    Ritorna:
        tupla (id_numerico, entiny_df_aggiornato)
    """
    if ref in ref_map:
        return ref_map[ref], entity_df

    new_id = int(entity_df[id_column_name].max()) + 1
    ref_map[ref] = new_id

    new_row = {id_column_name: new_id}
    if extra_fields:
        new_row.update(extra_fields)

    entity_df = pd.concat([entity_df, pd.DataFrame([new_row])], ignore_index=True)
    print(f"    Nuovo ID creato: {ref} -> {id_column_name}={new_id}")

    return new_id, entity_df


def check_coverage(seasons, driver_map, constructor_map):
    """
    Verifica quanti piloti/costruttori delle stagioni indicate sono già presenti nel dataset locale e quanti risulterebbero da creare

    Parametri:
        seasons: lista di anni da controllare
        driver_map, constructor_map: mappe ref->id già costruite

    Ritorna:
        dict con 'missing_drivers' e 'missing_constructors'
    """
    missing_drivers = set()
    missing_constructors = set()

    for season in seasons:
        # Prendiamo il primo round della stagione come campione rappresentativo
        # La formazione piloti/costruttori cambia raramente durante l'anno, un solo round basta a stimarla
        results = get_race_results(season, 1)
        for r in results:
            if r["driver_ref"] not in driver_map:
                missing_drivers.add(r["driver_ref"])
            if r["constructor_ref"] not in constructor_map:
                missing_constructors.add(r["constructor_ref"])

    return {"missing_drivers": missing_drivers, "missing_constructors": missing_constructors}


def build_new_rows(race, results, driver_map, constructor_map, circuit_map,
                    drivers_df, constructors_df, races_df, results_df):
    """
    Converte i dati di una gara scaricata da Jolpica nel formato delle tabelle locali CSV, creando nuovi ID per piloti/costruttori non ancora mappati.

    Parametri:
        race: dizionario con season, round, race_name, date, circuit_ref
        results: lista di dizionario da get_race_results()
        driver_map, constructor_map, circuit_map: mappe ref->id
        drivers_df, constructors_df, races_df, results_df: DataFrame
            correnti, a cui vengono eventualmente aggiunte nuove righe

    Ritorna:
        tupla (drivers_df, constructors_df, races_df, results_df, new_race_id)
        tutti aggiornati con le nuove righe
    """
    # Circuito: se non mappato, è un problema più serio di un driver/costruttore nuovo, lo segnaliamo chiaramente invece di crearlo silenziosamente
    if race["circuit_ref"] not in circuit_map:
        print(f"  ATTENZIONE: circuito '{race['circuit_ref']}' non mappato, gara saltata")
        return drivers_df, constructors_df, races_df, results_df, None

    circuit_id = circuit_map[race["circuit_ref"]]

    # Nuovo raceId sequenziale
    new_race_id = int(races_df["raceId"].max()) + 1

    # Nuova riga in races.csv
    new_race_row = {
        "raceId": new_race_id,
        "year": race["season"],
        "round": race["round"],
        "circuitId": circuit_id,
        "name": race["race_name"],
        "date": race["date"],
    }
    races_df = pd.concat([races_df, pd.DataFrame([new_race_row])], ignore_index=True)

    # Una riga in results.csv per ciascun pilota della gara
    new_result_id = int(results_df["resultId"].max()) + 1

    for r in results:
        driver_id, drivers_df = resolve_or_create_id(
            r["driver_ref"], driver_map, "driverId", drivers_df,
            extra_fields={"driverRef": r["driver_ref"]}
        )
        constructor_id, constructors_df = resolve_or_create_id(
            r["constructor_ref"], constructor_map, "constructorId", constructors_df,
            extra_fields={"constructorRef": r["constructor_ref"]}
        )

        # 'position' è NaN per i non classificati
        new_result_row = {
            "resultId": new_result_id,
            "raceId": new_race_id,
            "driverId": driver_id,
            "constructorId": constructor_id,
            "grid": r["grid"],
            "position": r["position_order"] if r["finished"] else None,
            "positionOrder": r["position_order"],
            "points": r["points"],
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_result_row])], ignore_index=True)
        new_result_id += 1

    return drivers_df, constructors_df, races_df, results_df, new_race_id


def main():
    print("Caricamento dati locali...")
    data = load_raw_data()
    driver_map, constructor_map, circuit_map = build_ref_maps(data)

    print("\nVerifica copertura piloti/costruttori...")
    coverage = check_coverage(SEASONS_TO_CHECK, driver_map, constructor_map)
    print(f"  Piloti non mappati: {coverage['missing_drivers'] or 'nessuno'}")
    print(f"  Costruttori non mappati: {coverage['missing_constructors'] or 'nessuno'}")

    print("\nRicerca gare nuove da scaricare...")
    existing = get_existing_races(data["races"])
    new_races = find_new_races(SEASONS_TO_CHECK, existing)
    print(f"  Gare da scaricare: {len(new_races)}")

    drivers_df = data["drivers"]
    constructors_df = data["constructors"]
    races_df = data["races"]
    results_df = data["results"]

    for race in new_races:
        print(f"\nScaricamento: {race['season']} round {race['round']} — {race['race_name']}")
        results = get_race_results(race["season"], race["round"])

        if not results:
            print("  Nessun risultato disponibile, salto.")
            continue

        drivers_df, constructors_df, races_df, results_df, new_id = build_new_rows(
            race, results, driver_map, constructor_map, circuit_map,
            drivers_df, constructors_df, races_df, results_df
        )
        if new_id:
            print(f"  Aggiunta come raceId={new_id}, {len(results)} risultati")

    # Backup dei CSV originali prima di sovrascrivere; rete di sicurezza in caso qualcosa vada storto durante la scrittura
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    backup_dir = os.path.join(data_dir, "backup_pre_update")
    os.makedirs(backup_dir, exist_ok=True)
    for fname in ["races.csv", "results.csv", "drivers.csv", "constructors.csv"]:
        src = os.path.join(data_dir, fname)
        if os.path.exists(src):
            pd.read_csv(src).to_csv(os.path.join(backup_dir, fname), index=False)

    print("\nSalvataggio CSV aggiornati...")
    races_df.to_csv(os.path.join(data_dir, "races.csv"), index=False)
    results_df.to_csv(os.path.join(data_dir, "results.csv"), index=False)
    drivers_df.to_csv(os.path.join(data_dir, "drivers.csv"), index=False)
    constructors_df.to_csv(os.path.join(data_dir, "constructors.csv"), index=False)

    print(f"\nCompletato. {len(new_races)} gare processate.")


if __name__ == "__main__":
    main()
