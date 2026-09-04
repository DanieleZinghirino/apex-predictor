"""
Client per l'API Jolpica-F1 (https://github.com/jolpica/jolpica-f1)

Jolpica è mantenuta da un piccolo gruppo di volontari, questo client rispetta i rate limit ufficiali per
buona educazione verso l'infrastruttura condivisa:
    - Burst limit: 4 richieste al secondo
    - Sustained limit: 500 richieste all'ora
(fonte: https://github.com/jolpica/jolpica-f1/blob/main/docs/rate_limits.md)
"""
import requests
import time

BASE_URL = "https://api.jolpi.ca/ergast/f1"

# Pausa tra una richiesta e l'altra: con 0.3s stiamo abbondantemente sotto il burst limit di 4/sec (che equivale a 0.25s tra richieste), lasciando un margine di sicurezza
REQUEST_DELAY_SECONDS = 0.3


def _get(endpoint, max_retries=3):
    """
    Esegue una richiesta GET verso l'API e ritorna il JSON parsato.
    Centralizza la pausa di rate-limiting: qualsiasi funzione di questo modulo che chiama _get eredita automaticamente il rispetto dei limiti

    Parametri:
        endpoint: percorso relativo a BASE_URL
        max_retries: numero di tentativi prima di sollevare l'errore

    Ritorna:
        dizionario con il JSON deserializzato della risposta

    Solleva:
        requests.RequestException se tutti i tentativi falliscono
    """
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", timeout=15)
            response.raise_for_status()
            time.sleep(REQUEST_DELAY_SECONDS)
            return response.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # backoff esponenziale: 1s, 2s, 4s
                print(f"    Errore di rete ({e.__class__.__name__}), ritento tra {wait}s...")
                time.sleep(wait)

    if last_error is not None:
        raise last_error

    raise RuntimeError(f"Richiesta fallita senza eccezione catturata per endpoint: {endpoint}")


def get_next_race_info():
    """
    Ritorna informazioni sulla prossima gara in calendario rispetto a oggi (stagione corrente, calcolata lato server da Jolpica).

    Ritorna:
        dizionario con season, round, race_name, date, circuit_ref oppure None se non ci sono gare future programmate
    """
    data = _get("current/next/")
    races = data["MRData"]["RaceTable"]["Races"]

    if not races:
        return None

    race = races[0]
    return {
        "season": race["season"],
        "round": race["round"],
        "race_name": race["raceName"],
        "date": race["date"],
        "circuit_ref": race["Circuit"]["circuitId"],
    }


def get_season_races(season):
    """
    Ritorna l'elenco di tutte le gare di una stagione, con round edata, usato per sapere QUALI round esistono prima di provare a scaricarne i risultati
    Parametri:
        season: anno

    Ritorna:
        Lista di dizionari con round, race_name, date, circuit_ref ordinata per round crescente
    """
    data = _get(f"{season}/races/")
    races = data["MRData"]["RaceTable"]["Races"]

    return [
        {
            "round": int(r["round"]),
            "race_name": r["raceName"],
            "date": r["date"],
            "circuit_ref": r["Circuit"]["circuitId"],
        }
        for r in races
    ]


def get_qualifying_results(season, round_number):
    """
    Ritorna la griglia di partenza per una gara specifica.

    Parametri:
        season: anno
        round_number: numero del round nella stagione

    Ritorna:
        Lista di dict con driver_ref, constructor_ref, grid_position ordinata per posizione in griglia, q1, q2, q3
        Lista vuota se le qualifiche non sono ancora state disputate
    """
    data = _get(f"{season}/{round_number}/qualifying/")
    races = data["MRData"]["RaceTable"]["Races"]

    if not races:
        return []

    return [
        {
            "driver_ref": entry["Driver"]["driverId"],
            "constructor_ref": entry["Constructor"]["constructorId"],
            "grid_position": int(entry["position"]),
            "q1": entry.get("Q1"),
            "q2": entry.get("Q2"),
            "q3": entry.get("Q3"),
        }
        for entry in races[0]["QualifyingResults"]
    ]


def get_race_results(season, round_number):
    """
    Ritorna i risultati di una gara già disputata.

    Parametri:
        season: anno
        round_number: numero del round nella stagione

    Ritorna:
        Lista di dict con driver_ref, constructor_ref, grid, position_order, points, finished. 
        Lista vuota se la gara non è ancora stata disputata.
        Include anche driver_forename/driver_surname, utili per creare righe complete quando un pilota non è ancora nel dataset locale

    Nota su 'finished': positionText nell'API Jolpica è un numero (posizione) se il pilota ha classificato, oppure un codice
    ("R"=ritirato, "D"=squalificato, "E"=escluso, "W"=ritirato prima del via, "F"=non qualificato, "N"=non partito) se non l'ha fatto.
    Questo replica la logica già usata su 'position.notnull()' nel dataset storico Kaggle, applicata qui ai dati live.
    """
    data = _get(f"{season}/{round_number}/results/")
    races = data["MRData"]["RaceTable"]["Races"]

    if not races:
        return []

    non_finish_codes = {"R", "D", "E", "W", "F", "N"}

    return [
        {
            "driver_ref": entry["Driver"]["driverId"],
            "driver_forename": entry["Driver"]["givenName"],
            "driver_surname": entry["Driver"]["familyName"],
            "constructor_ref": entry["Constructor"]["constructorId"],
            "constructor_name": entry["Constructor"]["name"],
            "grid": int(entry["grid"]),
            "position_order": int(entry["position"]),
            "points": float(entry["points"]),
            "finished": entry["positionText"] not in non_finish_codes,
        }
        for entry in races[0]["Results"]
    ]