"""
Client per Open-Meteo (https://open-meteo.com), API gratuita di dati meteo storici, usata per recuperare pioggia e temperatura nel giorno e luogo di ciascuna gara. 
Nessuna API key richiesta per uso non commerciale a basso volume.
"""
import requests
import time

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_race_weather(latitude, longitude, date_str):
    """
    Ritorna temperatura massima e precipitazioni totali per un località e data specifiche.

    Parametri:
        latitude, longitude: coordinate del circuito (da circuits.csv)
        date_str: data della gara, formato 'YYYY-MM-DD'

    Ritorna:
        dict con max_temp_c, precipitation_mm, oppure None se la richiesta fallisce o la data è troppo recente/futura per l'archivio storico
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "daily": "temperature_2m_max,precipitation_sum",
        "timezone": "auto",
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        time.sleep(0.2)  # gentile con l'API, anche se non ha rate limit dichiarati stringenti

        daily = data.get("daily", {})
        temps = daily.get("temperature_2m_max", [])
        precip = daily.get("precipitation_sum", [])

        if not temps or not precip:
            return None

        return {"max_temp_c": temps[0], "precipitation_mm": precip[0]}
    
    except requests.RequestException:
        return None