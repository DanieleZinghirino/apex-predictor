# Avvia il server dell'API Apex Predictor. Da lanciare dalla root del progetto, con il virtual environment attivo.
echo "=========================================="
echo "Apex Predictor API"
echo "=========================================="
echo "Interfaccia web disponibile su: http://localhost:8000"
echo "Documentazione API interattiva: http://localhost:8000/docs"
echo "Premi Ctrl+C per fermare il server"
echo "=========================================="
echo ""
uvicorn api.main:app --port 8000