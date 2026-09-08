# Aggiorna dati e modello, poi avvia il server con il modello nuovo
set -e

echo "Aggiornamento dati e riallenamento modello..."
python3 scripts/update_data.py
python3 scripts/backfill_qualifying.py
python3 scripts/fetch_weather.py
python3 run_pipeline.py

echo ""
echo "Avvio server con il modello aggiornato..."
./scripts/start_api.sh