"""
API REST per Apex Predictor.

Espone la logica di previsione già validata in src/ come servizio HTTP, consumato sia dalla pagina web statica (api/static/index.html) sia dal bot 
Telegram (bot/telegram_bot.py), nessuna logica duplicata tra i due client.
"""
import sys
from contextlib import asynccontextmanager
import threading
import os
# Aggiunge la cartella superiore (la root del progetto) al path di ricerca di Python
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.data_loading import load_raw_data, build_working_dataset, PROJECT_ROOT
from src.jolpica_client import get_next_race_info, get_qualifying_results
from src.live_predict import map_refs_to_ids, build_upcoming_race_features, build_pre_qualifying_features
from src.predict import load_trained_model, predict_podium


def run_telegram_bot():
    """
    Avvia il bot Telegram in un thread separato. Eventuali eccezioni
    vengono stampate esplicitamente, altrimenti spariscono silenziosamente
    nel thread daemon.
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    try:
        from bot.telegram_bot import main as bot_main
        bot_main()
    except Exception as e:
        print(f"ERRORE: il bot Telegram si è fermato con un'eccezione: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Sostituisce il vecchio @app.on_event("startup"), rimosso nelle
    versioni più recenti di Starlette/FastAPI (da cui dipendevamo
    senza fissare la versione in requirements-deploy.txt). Tutto ciò
    che sta PRIMA di 'yield' gira all'avvio del server; ciò che sta
    DOPO (qui assente) girerebbe allo spegnimento.
    """
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        thread = threading.Thread(target=run_telegram_bot, daemon=True)
        thread.start()
        print("Bot Telegram avviato in background.")
    else:
        print("TELEGRAM_BOT_TOKEN non impostato, bot non avviato.")
    yield  # il server gira qui, tra startup e shutdown


# FastAPI() crea l'applicazione vera e propria, l'oggetto a cui agganciamo tutte le rotte (gli URL) che il server saprà gestire.
app = FastAPI(title="Apex Predictor API", version="1.0", lifespan=lifespan)

# CORS = Cross-Origin Resource Sharing. I browser, per sicurezza, bloccano di default le richieste JavaScript verso un dominio diverso da quello che ha servito la pagina
# allow_origins=["*"] disabilita questo controllo per qualisasi origine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Il modello viene caricato qui, a livello di modulo
# MODEL e THRESHOLD restano poi disponibili a tutte le funzioni sotto senza doverli ricaricare
print("Caricamento modello all'avvio del server...")
MODEL, THRESHOLD = load_trained_model()
print("Modello caricato.")


# Il decoratore @app.get(...) indica una funzione di callback all'arrivo di una richiesta HTTP GET su questo URL. 
# Il nome della funzione (health_check) non ha alcun effetto sull'URL, è solo per leggibilità del codice
@app.get("/health")
def health_check():
    """
    Endpoint minimo per verificare che il server sia vivo
    """
    return {"status": "ok"}


@app.get("/predict/next-race")
def predict_next_race():
    """
    Genera la previsione di podio per la prossima gara F1 in calendario. Stessa identica logica di scripts/predict_next_race.py, qui restituita come JSON
    invece che stampata a schermo

    Ritorna:
        Un dizionario Python, che FastAPI converte automaticamente in JSON prima di rispondere
    """
    race_info = get_next_race_info()

    if race_info is None:
        # HTTPException è il modo corretto di segnalare un errore in FastAPI: invece di un errore Python generico, il client riceve una risposta HTTP con status 
        # code 404 e il messaggio nel corpo
        raise HTTPException(status_code=404, detail="Nessuna gara futura in calendario")

    data = load_raw_data()
    historical_df = build_working_dataset(data["races"], data["results"], min_year=2004)

    circuit_map = dict(zip(data["circuits"]["circuitRef"], data["circuits"]["circuitId"]))
    circuit_id = circuit_map.get(race_info["circuit_ref"])

    if circuit_id is None:
        if not (race_info.get("circuit_lat") and race_info.get("circuit_lng")):
            raise HTTPException(status_code=422, detail=f"Circuito '{race_info['circuit_ref']}' non risolvibile")
        new_id = int(data["circuits"]["circuitId"].max()) + 1
        new_row = {
            "circuitId": new_id, "circuitRef": race_info["circuit_ref"],
            "name": race_info["circuit_name"], "location": race_info.get("circuit_country", ""),
            "country": race_info.get("circuit_country", ""),
            "lat": float(race_info["circuit_lat"]), "lng": float(race_info["circuit_lng"]),
        }
        data["circuits"] = pd.concat([data["circuits"], pd.DataFrame([new_row])], ignore_index=True)
        data["circuits"].to_csv(os.path.join(PROJECT_ROOT, "data", "raw", "circuits.csv"), index=False)
        circuit_id = new_id

    qualifying = get_qualifying_results(race_info["season"], race_info["round"])

    if qualifying:
        # Griglia reale disponibile: previsione "definitiva"
        qualifying_df = map_refs_to_ids(qualifying, data["drivers"], data["constructors"])
        features_df = build_upcoming_race_features(qualifying_df, historical_df, circuit_id, data["circuits"])
        is_definitive = True
    else:
        # Griglia non ancora nota: previsione "anticipata" con stima
        features_df = build_pre_qualifying_features(
            historical_df, circuit_id, int(race_info["season"]), data["circuits"]
        )
        is_definitive = False

    # MODEL e THRESHOLD sono le variabili globali caricate una volta sola all'avvio
    predictions = predict_podium(MODEL, THRESHOLD, features_df)
    predictions = predictions.merge(data["drivers"][["driverId", "surname"]], on="driverId", how="left")
    predictions = predictions.sort_values("podium_probability", ascending=False)

    weather = None
    if "race_max_temp_c" in features_df.columns and not features_df.empty:
        weather = {
            "temp_c": round(float(features_df["race_max_temp_c"].iloc[0]), 1),
            "precipitation_mm": round(float(features_df["race_precipitation_mm"].iloc[0]), 1),
        }

    # Costruiamo qui la struttura JSON finale che il client riceverà.
    # Nota i cast espliciti float()/bool(): i tipi numpy non sono serializzabili in JSON di default, vanno convertiti ai tipi Python nativi
    # (float, bool, int) prima di restituirli, altrimenti FastAPI solleverebbe un errore di serializzazione
    return {
        "race_name": race_info["race_name"],
        "date": race_info["date"],
        "is_definitive": is_definitive,
        "weather": weather,
        "predictions": [
            {
                "driver": row["surname"],
                "grid": round(float(row["grid"]), 1),
                "probability": round(float(row["podium_probability"]), 4),
                "predicted_podium": bool(row["podium_predicted"]),
            }
            for _, row in predictions.iterrows()
        ],
    }

# StaticFiles serve file statici (HTML, CSS, immagini) da una cartella del disco, senza bisogno di scrivere una funzione dedicata per ognuno.
# html=True cerca e serve automaticamente un index.html
# ATTENZIONE ALL'ORDINE: questa riga va DOPO tutte le @app.get(...) di cui sopra.
# FastAPI controlla le rotte nell'ordine in cui sono definite nel codice,se il mount fosse prima, intercetterebbe anche le richieste a /predict/next-race
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")