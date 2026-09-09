# Apex Predictor

Sistema di machine learning che prevede la probabilità che un pilota di Formula 1 arrivi sul podio (top 3) o vinca la gara, usando dati storici (2004-2026, aggiornati automaticamente da fonte live) e due modelli XGBoost paralleli, con 14 feature selezionate dopo un'analisi rigorosa di importanza (gain, permutation importance, SHAP) e confrontate con 4 architetture di deep learning.

## Usalo subito

- **Pagina web**: [apex-predictor.onrender.com](https://apex-predictor.onrender.com), apri il link, clicca il bottone, ottieni la previsione
- **Bot Telegram**: [@f1podiumbot](https://t.me/f1podiumbot), invia `/podium` per la probabilità di podio, `/victory` per la probabilità di vittoria

Nessun account, nessun setup, gratuito. Il servizio gratuito si "addormenta" dopo ~15 minuti di inattività: la prima richiesta dopo una pausa può richiedere 30-60 secondi in più per il risveglio.

## Come funziona, in breve

1. Dati storici F1 (Kaggle, 2004-2024) + aggiornamento automatico via API (Jolpica-F1, 2025-oggi)
2. Feature engineering senza data leakage temporale, con media mobile esponenziale per la forma recente e mediana robusta per la stima della griglia
3. Due modelli XGBoost paralleli, **podio** (top 3) e **vincitore**, validati con `TimeSeriesSplit`, calibrati per **precision alta** (il progetto genera previsioni pubbliche, dove un falso allarme è visibile e costa credibilità)
4. Le probabilità mostrate sono **quote relative**: sommano al 100% sull'intero gruppo di piloti in gara, invece di probabilità assolute indipendenti
5. Previsione sulla prossima gara reale, con griglia e meteo effettivi quando disponibili, stimati altrimenti, circuiti debuttanti gestiti automaticamente (creazione da coordinate Jolpica + tentativo di recupero caratteristiche da Wikipedia)

## Guida rapida, interfacce disponibili

### Pagina web
Apri [apex-predictor.onrender.com](https://apex-predictor.onrender.com), clicca **"Genera previsione prossima gara"**. La tabella mostra, per ciascun pilota: griglia (reale o stimata), quota podio con evidenziazione (🟢 primi 3, 🔴 altri sopra soglia), quota vittoria.

### Bot Telegram
Cerca **@f1podiumbot** su Telegram, avvia la chat:
- `/podium` → quote di podio per tutti i piloti
- `/victory` → quote di vittoria, ordinate per probabilità

### API REST (per sviluppatori)

GET https://apex-predictor.onrender.com/predict/next-race

Documentazione interattiva: `https://apex-predictor.onrender.com/docs`

## Eseguirlo in locale (per contribuire o modificare il codice)

```bash
git clone <url-repo>
cd apex-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Workflow completo, un comando:**
```bash
./scripts/run_all.sh
```
Aggiorna storico (Jolpica), backfill qualifiche, meteo, riallena entrambi i modelli, genera la previsione. Si interrompe al primo errore.

**Script individuali:**
```bash
./scripts/download_data.sh          # dataset storico Kaggle (una tantum)
python3 scripts/update_data.py       # nuove gare da Jolpica-F1
python3 scripts/backfill_qualifying.py  # tempi di qualifica mancanti
python3 scripts/fetch_weather.py     # meteo storico (Open-Meteo)
python3 run_pipeline.py              # training completo (podio + vincitore)
python3 scripts/predict_next_race.py # previsione da terminale
python3 try_predictions.py           # demo su una gara già disputata
```

**Interfaccia web in locale:**
```bash
./scripts/start_api.sh                # avvio rapido
./scripts/update_and_serve.sh         # aggiorna dati/modello, poi avvia
```
Il modello si carica in memoria solo all'avvio, dopo un `run_all.sh`, riavvia il server per usare il modello aggiornato.

**Bot Telegram in locale:**
1. Crea un bot con [@BotFather](https://t.me/BotFather), ottieni il token
2. `.env` nella root: `TELEGRAM_BOT_TOKEN=il-tuo-token`
3. `python3 bot/telegram_bot.py` (con l'API già in esecuzione)

**Attenzione**: non far girare il bot in locale mentre quello su Render è attivo, Telegram permette un solo processo di polling per token, altrimenti si genera un conflitto che blocca entrambi.

## Deploy (Render, piano gratuito)

Configurazione in `render.yaml`. Build: `pip install -r requirements-deploy.txt` (dipendenze minime, senza tooling da notebook). Start: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`. Il bot Telegram gira come thread in background nello stesso processo dell'API (necessario sul piano gratuito, che offre un solo servizio web). Variabile d'ambiente da impostare nella dashboard: `TELEGRAM_BOT_TOKEN`.

## Feature dei modelli (14, dopo selezione)

Un'analisi sistematica (gain XGBoost, permutation importance corretta per la soglia operativa, SHAP values) ha confrontato 22 feature iniziali, scartandone 8 con contributo trascurabile, verificate una per una per escludere bug di implementazione prima di rimuoverle.

**Piloti/scuderie**: forma recente (media mobile esponenziale), affidabilità scuderia, storico su circuito, confronto col compagno di squadra, posizione in classifica generale.

**Circuito**: numero curve, velocità media storica, indice di sorpassabilità (calcolati dai dati reali).

**Qualifica**: distacco dal poleman in secondi.

**Meteo**: temperatura massima, precipitazioni.

## Risultati

Validati su gare mai viste in training (2025 e parte del 2026):

| Modello | Precision | Recall | F1 | Soglia |
|---|---|---|---|---|
| Podio (top 3) | 0.636 | 0.820 | 0.717 | 0.70 |
| Vincitore | 0.650 | 0.703 | 0.675 | 0.85 |

Il modello vincitore ha uno sbilanciamento molto più estremo (~5% di positivi contro ~15% del podio), coerente con la soglia più severa e l'F1 più basso.

### Evoluzione del modello podio

| Fase | Precision | Recall | F1 |
|---|---|---|---|
| XGBoost tunato, test 2023-2024 | 0.628 | 0.746 | 0.682 |
| + backfill storico 2025-2026 | 0.629 | 0.833 | 0.717 |
| + feature circuiti | 0.650 | 0.824 | 0.727 |
| + 22 feature totali | 0.624 | 0.861 | 0.724 |
| **14 feature (dopo SHAP), retunato** | **0.636** | **0.820** | **0.717** |

### Confronto con il deep learning

4 architetture PyTorch testate con la stessa metodologia (split temporale, ricerca soglia su F1): MLP con embedding categorici, LSTM sulla sequenza grezza di gare, TabNet, GNN sulle relazioni compagno di squadra.

| Modello | F1 |
|---|---|
| **XGBoost (campione)** | **0.717** |
| TabNet | 0.711 |
| LSTM ibrida | 0.678 |
| GNN (GraphSAGE) | 0.667 |
| MLP + embedding | 0.657 |

Nessuna rete ha superato XGBoost, risultato coerente con la letteratura: con ~8500 righe di training, gradient boosting resta generalmente superiore alle reti neurali su dati tabellari. TabNet, il più vicino, ha mostrato un ranking di importanza feature quasi identico a quello SHAP di XGBoost, buona conferma incrociata che il segnale dominante (griglia, forma recente, classifica) è reale. Dettaglio in `notebooks/06`-`10`.

## Metodologia e decisioni chiave

**Range dati**: 2004-2024 (Kaggle) esteso a 2025-2026 (Jolpica-F1). Il tracciamento del giro veloce, introdotto nel 2004, è risultato fortemente predittivo, motivando l'esclusione degli anni precedenti.

**Nessun data leakage temporale**: ogni feature dinamica usa solo dati precedenti alla gara prevista, verificato manualmente su casi singoli (es. Verstappen a Zandvoort per la gara di casa). Standings presi dalla gara precedente, mai da quella corrente.

**Selezione modello**: confrontati 8 algoritmi classici, ottimizzati i due migliori con `RandomizedSearchCV` + `TimeSeriesSplit`, poi 4 architetture di deep learning. XGBoost resta il campione in ogni confronto.

**Permutation importance, lezione metodologica**: instabile con feature correlate (permutarne una sottostima il contributo se un'altra copre il segnale); SHAP ha dato un quadro più affidabile per la selezione finale delle feature.

**Quote relative invece di probabilità assolute**: le probabilità individuali del modello vengono normalizzate per sommare a 100% sull'intero gruppo, più intuitivo da leggere ("chi ha più chance rispetto agli altri"), con soglia di classificazione riscalata in modo matematicamente equivalente a quella calibrata in training.

**Griglia stimata robusta agli outlier**: per le previsioni anticipate, la griglia è la mediana (non la media) delle ultime 6 gare, una singola penalità o griglia anomala non distorce la stima (caso reale verificato con Antonelli dopo Monza 2026).

## Limiti noti

- **Precipitazione giornaliera, non oraria**: può sovrastimare le gare "bagnate"
- **Circuiti debuttanti senza storico specifico**: le feature calcolate dallo storico (velocità media, sorpassi) ricadono sulla media globale; le caratteristiche fisiche (curve) si ottengono da scraping Wikipedia best-effort, non garantito
- **Un singolo DNF isolato** può non essere pienamente pesato dalla forma generale se le gare precedenti erano forti
- **Modello vincitore meno accurato** del modello podio, per lo sbilanciamento di classe più estremo intrinseco al problema
- **Piano di hosting gratuito**: il servizio si addormenta dopo inattività; il bot Telegram può generare conflitti di polling se girano contemporaneamente un'istanza locale e quella su Render

## Struttura del progetto

apex-predictor/
├── api/
│ ├── main.py # API REST (FastAPI) + avvio bot in background
│ └── static/index.html # interfaccia web
├── bot/
│ └── telegram_bot.py # bot Telegram (python-telegram-bot)
├── data/
│ ├── raw/ # dati Kaggle + backfill Jolpica (versionati, essenziali)
│ └── reference/ # dati curati dal progetto (circuiti, meteo)
├── docs/ # prompt LLM per dati circuiti, documentazione di processo
├── models/ # modelli podio + vincitore, versionati per il deploy
├── notebooks/ # 01-05 EDA/feature/tuning/analisi, 06-10 deep learning
├── src/ # codice riutilizzabile e testato
│ ├── data_loading.py, features.py, train.py, predict.py, dl_common.py
│ ├── live_predict.py # feature per gare future
│ └── jolpica_client.py, weather_client.py, circuit_scraper.py
├── scripts/ # script eseguibili standalone
├── render.yaml # configurazione deploy
├── requirements.txt # ambiente di sviluppo completo
├── requirements-deploy.txt # dipendenze minime per il deploy
└── README.md


## Sviluppi futuri possibili

- **Flag DNF ultima gara**: catturare esplicitamente un incidente/ritiro recente isolato, non pienamente pesato dalla forma media
- **Meteo orario** invece che giornaliero, per una stima più precisa di "pioggia durante la gara"
- **Automazione completa delle caratteristiche circuito** per debutti assoluti, oltre allo scraping Wikipedia best-effort attuale
- **Retuning degli iperparametri sul set di 14 feature** (attualmente riusa i parametri ottimizzati sul set a 22) per un'ultima rifinitura del modello podio
- **Intervalli di confidenza** sulle previsioni (regressione quantile o conformal prediction), specialmente utili su circuiti debuttanti dove l'incertezza è strutturalmente più alta
- **Notifiche automatiche**: il bot invia la previsione in autonomia ogni weekend di gara, invece di attendere il comando dell'utente
- **Estensione ad altre categorie** (F2, IndyCar) riusando la stessa architettura

## Stato del progetto

✅ **Completo e pubblicamente accessibile.** Pipeline end-to-end, due modelli predittivi validati e confrontati con 4 architetture di deep learning, interfaccia web e bot Telegram in produzione.