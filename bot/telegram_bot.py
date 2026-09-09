"""
Bot Telegram per Apex Predictor.

Framework: python-telegram-bot (PTB), diverso da FastAPI per paradigma.
FastAPI è request/response: il client chiede, il server risponde una volta. 
PTB è event-driven: il bot resta in ascolto continuo di eventi e reagisce ad ognuno in modo indipendente
"""
import os
import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Carica le variabili da .env
load_dotenv()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
# URL dell'API FastAPI che il bot interroga
_default_port = os.environ.get("PORT", "8000")
API_URL = os.environ.get("API_URL", f"http://localhost:{_default_port}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gestisce il comando /start, il primo messaggio che un utente Telegram vede aprendo una chat col bot per la prima volta.

    Parametri:
        update: contiene i dettagli del messaggio ricevuto
        context: strumenti per rispondere, dati condivisi tra comandi
    """
    message = update.message
    if message is None:
        return

    await message.reply_text(
        "Benvenuto su Apex Predictor!\n\n"
        "Usa /podium per la probabilità di arrivare a podio.\n"
        "Usa /victory per la probabilità di vincere la gara."
    )


async def _fetch_prediction(update: Update):
    """
    Funzione condivisa tra /podium e /victory: chiama l'API e ritorna il JSON della previsione, oppure None se la chiamata fallisce.
    Evita di duplicare la logica di rete e gestione errori in entrambi i comandi.

    Parametri:
        update: contiene i dettagli del messaggio ricevuto, serve per poter rispondere all'utente in caso di errore

    Ritorna:
        dict con i dati della previsione, oppure None in caso di errore
    """
    message = update.message
    if message is None:
        return None

    # Messaggio immediato di attesa, perché la chiamata all'API può richiedere qualche secondo
    await message.reply_text("Sto calcolando la previsione...")

    try:
        # httpx.AsyncClient invece di requests: PTB è asincrono, httpx ha una versione async nativa che si integra bene a differenza di requests
        # che bloccherebbe l'intero bot durante l'attesa
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_URL}/predict/next-race")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        await message.reply_text(
            f"Errore nel contattare il servizio di previsione: {e}\n"
            f"Riprova tra qualche minuto."
        )
        return None


def _build_header(data):
    """
    Costruisce le righe di intestazione comuni a /podium e /victory:
    nome gara, data, tipo di previsione (definitiva/anticipata), meteo.
    Tenerla separata evita di duplicare questa logica in entrambi i comandi.

    Parametri:
        data: il JSON della previsione ritornato dall'API

    Ritorna:
        lista di stringhe, una per riga di intestazione
    """
    label = "DEFINITIVA" if data["is_definitive"] else "ANTICIPATA (griglia stimata)"

    lines = [
        f"🏁 {data['race_name']}",
        f"📅 {data['date']}",
        f"Previsione: {label}",
    ]

    if data.get("weather"):
        w = data["weather"]
        wet_label = "pioggia probabile" if w["precipitation_mm"] > 1 else "asfalto asciutto"
        lines.append(f"🌡️ {w['temp_c']}°C, {wet_label}")

    return lines


async def podium(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gestisce il comando /podium: 
    mostra solo la probabilità di arrivare a podio (top 3), ordinata per quota podio decrescente
    """
    data = await _fetch_prediction(update)
    if data is None:
        return

    message = update.message
    if message is None:
        return

    lines = _build_header(data)
    lines.append("")  # riga vuota come separatore visivo
    lines.append("Probabilità di arrivare a podio:")
    lines.append("")

    # Mostriamo solo i primi 8 piloti per non superare il limite di lunghezza dei messaggi Telegram (4096 caratteri) e per leggibilità
    for i, p in enumerate(data["predictions"][:5]):
        if i < 3:
            marker = "🏆"  # primi 3 per probabilità: podio più probabile
        elif p["predicted_podium"]:
            marker = "🔴"  # oltre il 3° ma comunque sopra soglia
        else:
            marker = "▫️"
        pct = round(p["podium_share"], 1)
        lines.append(f"{marker} {p['driver']}) Partenza: {int(p['grid'])}; Prob. podio: {pct}%")

    await message.reply_text("\n".join(lines))


async def victory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gestisce il comando /victory:
    mostra solo la probabilità di vincere la gara.
    """
    data = await _fetch_prediction(update)
    if data is None:
        return

    message = update.message
    if message is None:
        return

    lines = _build_header(data)
    lines.append("")
    lines.append("Probabilità di vincere la gara:")
    lines.append("")

    sorted_by_winner = sorted(data["predictions"], key=lambda p: p["winner_share"], reverse=True)

    for i, p in enumerate(sorted_by_winner[:5]):
        marker = "🥇" if i == 0 else "🔴"
        pct = round(p["winner_share"], 1)
        lines.append(f"{marker} {p['driver']}) Partenza: {int(p['grid'])}; Prob. vittoria: {pct}%")

    await message.reply_text("\n".join(lines))


def main():
    """
    Punto di ingresso: costruisce l'applicazione del bot, registra quali funzioni gestiscono quali comandi, e lo mette in ascolto
    permanente (polling) finché non viene interrotto manualmente.
    """
    if not TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN non trovato. Crea un file .env nella root del progetto con: TELEGRAM_BOT_TOKEN=il-tuo-token"
        )

    # Application è l'oggetto centrale di PTB, equivalente concettuale dell'oggetto FastAPI in api/main.py, a cui agganciamo gli handler
    app = Application.builder().token(TOKEN).build()

    # CommandHandler collega ciascun comando digitato dall'utente alla funzione corrispondente; stesso concetto del decoratore
    # @app.get(...) in FastAPI, sintassi diversa perché framework diverso
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("podium", podium))
    app.add_handler(CommandHandler("victory", victory))

    print("Bot avviato. In ascolto di messaggi... (Ctrl+C per fermare)")
    # run_polling(): il bot interroga continuamente i server Telegram chiedendo "ci sono nuovi messaggi per me?"
    app.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()