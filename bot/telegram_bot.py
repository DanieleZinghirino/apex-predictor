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
        "Usa /run per ricevere la previsione di podio per la prossima gara di Formula 1"
    )


async def previsione(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Gestisce il comando /run — chiama l'API e formatta il risultato in un messaggio di testo leggibile per Telegram
    """
    # Messaggio immediato di attesa, perché la chiamata all'API può richiedere qualche secondo
    message = update.message
    if message is None:
        return

    await message.reply_text("Sto calcolando la previsione...")

    try:
        # httpx.AsyncClient invece di requests: PTB è asincrono, httpx ha una versione async nativa che si integra bene a differenza di requests 
        # che bloccherebbe l'intero bot durante l'attesa
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{API_URL}/predict/next-race")
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as e:
        await message.reply_text(
            f"Errore nel contattare il servizio di previsione: {e}\n"
            f"Riprova tra qualche minuto."
        )
        return

    label = "DEFINITIVA" if data["is_definitive"] else "ANTICIPATA (griglia stimata)"

    # Costruiamo il messaggio come testo semplice
    lines = [
        f"🏁 {data['race_name']}",
        f"📅 {data['date']}",
        f"Previsione: {label}",
    ]

    if data.get("weather"):
        w = data["weather"]
        wet_label = "pioggia probabile" if w["precipitation_mm"] > 1 else "asfalto asciutto"
        lines.append(f"🌡️ {w['temp_c']}°C, {wet_label}")

    lines.append("")  # riga vuota come separatore visivo

    # Il favorito per la vittoria, in evidenza separata prima della lista podio
    top_winner = max(data["predictions"], key=lambda p: p["winner_share"])
    lines.append(f"🥇 Favorito vittoria: {top_winner['driver']} ({top_winner['winner_share']}%)")
    lines.append("")

    # Mostriamo solo i primi 8 piloti per non superare il limite di lunghezza dei messaggi Telegram (4096 caratteri) e per leggibilità
    for i, p in enumerate(data["predictions"][:6]):
        if i < 3:
            marker = "🏆"  # primi 3 per probabilità: podio più probabile
        else:
            marker = "▫️"
        pct = round(p["podium_share"], 1)
        lines.append(f"{marker} {p['driver']} — griglia {p['grid']} — {pct}%")

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
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", previsione))  # o "previsione", usa il nome esatto della tua funzione


    # CommandHandler collega il comando "/run" digitato dall'utente alla funzione previsione()
    # definita sopra; stesso concetto del decoratore @app.get(...) in FastAPI
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("run", previsione))

    print("Bot avviato. In ascolto di messaggi... (Ctrl+C per fermare)")
    # run_polling(): il bot interroga continuamente i server Telegram
    app.run_polling(stop_signals=None)


if __name__ == "__main__":
    main()