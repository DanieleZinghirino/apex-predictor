# Esegue l'intero workflow di Apex Predictor in sequenza: aggiorna lo storico, riallena il modello, genera la previsione per la prossima gara
# Un comando solo, dalla root del progetto.
set -e  # interrompe tutto al primo errore, invece di proseguire con dati parziali

echo "=========================================="
echo "1/5 — Aggiornamento storico gare (Jolpica)"
echo "=========================================="
python3 scripts/update_data.py

echo ""
echo "=========================================="
echo "2/5 — Backfill qualifiche mancanti"
echo "=========================================="
python3 scripts/backfill_qualifying.py

echo ""
echo "=========================================="
echo "3/5 — Aggiornamento dati meteo storici"
echo "=========================================="
python3 scripts/fetch_weather.py

echo ""
echo "=========================================="
echo "4/5 — Riallenamento del modello"
echo "=========================================="
python3 run_pipeline.py

echo ""
echo "=========================================="
echo "5/5 — Previsione prossima gara"
echo "=========================================="
python3 scripts/predict_next_race.py

echo ""
echo "=========================================="
echo "Completato: dati aggiornati, modello riallenato, previsione generata."
echo "=========================================="