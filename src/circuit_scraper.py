"""
Estrazione best-effort di caratteristiche circuito dall'infobox Wikipedia. 
NON garantita: il formato delle infobox varia da pagina a pagina, e non tutti i circuiti hanno questi campi strutturati allo stesso modo. 
"""
import requests
import re


def scrape_circuit_characteristics(wiki_url):
    """
    Tenta di estrarre lunghezza (km) e numero di curve dall'infobox di una pagina Wikipedia del circuito.

    Parametri:
        wiki_url: URL della pagina Wikipedia del circuito

    Ritorna:
        dict con 'length_km' e 'num_corners'
    """
    result: dict[str, float | int | None] = {"length_km": None, "num_corners": None}

    try:
        response = requests.get(wiki_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        html = response.text
    except requests.RequestException:
        return result

    length_match = re.search(r"Length[^0-9]{0,40}?([\d.]+)\s*km", html, re.IGNORECASE)
    if length_match:
        try:
            result["length_km"] = float(length_match.group(1))
        except ValueError:
            pass

    corners_match = re.search(r"(?:Turns|Corners)[^0-9]{0,20}?(\d+)", html, re.IGNORECASE)
    if corners_match:
        try:
            result["num_corners"] = int(corners_match.group(1))
        except ValueError:
            pass

    return result