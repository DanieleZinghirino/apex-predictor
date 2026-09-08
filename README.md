# Apex Predictor

Sistema di machine learning che prevede la probabilità che un pilota di Formula 1 finisca sul podio (top 3), usando dati storici (2004-2026, aggiornati automaticamente da fonte live) e un modello XGBoost, con 14 feature selezionate dopo un'analisi rigorosa di importanza

## Come funziona, in breve

1. Dati storici F1 (Kaggle, 2004-2024) + aggiornamento automatico via API (Jolpica-F1, 2025-oggi)
2. Feature engineering senza data leakage temporale, con media mobile esponenziale per pesare la forma recente
3. Modello XGBoost, validato con `TimeSeriesSplit`, calibrato per **precision alta**
4. Previsione sulla prossima gara reale, con griglia e meteo effettivi quando disponibili, stimati altrimenti; **circuiti debuttanti gestiti automaticamente**

## Setup

```bash
git clone <url-repo>
cd apex-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow, un comando solo

```bash
./scripts/run_all.sh
```

Esegue in sequenza: aggiornamento storico (Jolpica), backfill qualifiche mancanti, aggiornamento meteo, riallenamento del modello, previsione per la prossima gara. Si interrompe al primo errore invece di proseguire con dati incompleti.

**Script individuali**, se serve eseguirli separatamente:
```bash
./scripts/download_data.sh          # dataset storico Kaggle (una tantum)
python3 scripts/update_data.py           # nuove gare da Jolpica-F1
python3 scripts/backfill_qualifying.py   # tempi di qualifica mancanti
python3 scripts/fetch_weather.py         # meteo storico (Open-Meteo)
python3 run_pipeline.py                  # training completo
python3 scripts/predict_next_race.py     # previsione prossima gara
python3 try_predictions.py               # demo su una gara già disputata

## Interfaccia web

Oltre agli script da terminale, il progetto espone un'API REST (FastAPI) con una pagina web che genera la previsione con un click.

**Avvio rapido (dati/modello già aggiornati):**
```bash
./scripts/start_api.sh
```
Apri `http://localhost:8000` nel browser.

**Avvio con aggiornamento completo (dati, modello, poi server):**
```bash
./scripts/update_and_serve.sh
```

**Importante**: il modello viene caricato in memoria una sola volta, all'avvio del server. Se viene aggiornato dati/modello (`run_all.sh`) mentre il server è già acceso, bisogna riavviarlo (Ctrl+C, poi `./scripts/start_api.sh`) per usare il modello aggiornato, altrimenti continuerà a servire previsioni con quello vecchio.

**Documentazione interattiva dell'API**: `http://localhost:8000/docs`, generata automaticamente da FastAPI, permette di testare l'endpoint `/predict/next-race` direttamente dal browser senza passare dalla pagina web.
```

## Feature del modello (14, dopo selezione)

Un'analisi sistematica (gain XGBoost, permutation importance, SHAP values) ha confrontato 22 feature iniziali, portando a scartarne 8 con contributo trascurabile confermato.

**Piloti/scuderie**: forma recente, affidabilità scuderia, storico su circuito, confronto col compagno di squadra, posizione in classifica generale (pilota e costruttore).

**Circuito**: numero curve, velocità media storica, indice di sorpassabilità (calcolati dai dati reali, non stimati).

**Qualifica**: distacco dal poleman in secondi (Q1/Q2/Q3).

**Meteo**: temperatura massima, precipitazioni (Open-Meteo, archivio storico per il training, previsione per le gare future).

## Risultati del modello

Validato su gare mai viste in training (2025 e parte del 2026):

| Metrica (classe Podio) | Valore |
|---|---|
| Precision | 0.636 |
| Recall | 0.820 |
| F1-score | 0.717 |
| Accuracy | 0.91 |
| Soglia | 0.70 |

### Evoluzione del modello

| Fase | Precision | Recall | F1 | Note |
|---|---|---|---|---|
| XGBoost tunato, test 2023-2024 | 0.628 | 0.746 | 0.682 | Modello base |
| + backfill storico 2025-2026 | 0.629 | 0.833 | 0.717 | Più dati reali |
| + feature circuiti | 0.650 | 0.824 | 0.727 | +6 feature |
| + gara di casa, standings, compagno, qualifica, meteo (22 feature) | 0.614 | 0.847 | 0.712 | Bug: qualifiche 2025-26 non ancora scaricate |
| Fix copertura dati qualifica | 0.624 | 0.861 | 0.724 | Dato reale invece di fallback |
| **14 feature (dopo analisi SHAP), retunato** | **0.636** | **0.820** | **0.717** | Modello più snello, pari performance |

## Metodologia e decisioni chiave

**Range dati: 2004-2024**, poi esteso a 2025-2026 via Jolpica-F1. Il tracciamento del giro veloce (dal 2004) è risultato fortemente predittivo, motivando questa scelta.

**Target su `positionOrder`**, non `position` (NaN per i ritiri).

**Nessun data leakage temporale**: ogni feature dinamica usa solo dati precedenti alla gara prevista, verificato manualmente su casi singoli. Standings presi dalla gara precedente nello stesso anno.

**Selezione modello**: confrontati 8 algoritmi, ottimizzati i due migliori con `RandomizedSearchCV` + `TimeSeriesSplit`. XGBoost è il campione, calibrato per precision alta.

**Analisi di importanza feature, una lezione metodologica**: la permutation importance "naive" (soglia di default 0.5) dava risultati fuorvianti, perché il modello opera con soglia 0.70, corretto ricalcolandola con la soglia operativa reale. Anche corretta, la permutation importance è risultata instabile in presenza di feature fortemente correlate: permutarne una alla volta sottostima il suo contributo, perché le altre coprono il buco. **SHAP values** ha dato un quadro più stabile e affidabile, usato per la decisione finale di quali feature scartare.

**Verifica pre-rimozione**: prima di scartare le 8 feature deboli, ciascuna è stata controllata singolarmente per escludere che il basso contributo fosse dovuto a un bug di implementazione piuttosto che a segnale genuinamente assente.

## Limiti noti, documentati onestamente

**Precipitazione giornaliera, non oraria**: `race_is_wet` si basa sul totale di pioggia nell'intera giornata, non specificamente durante l'orario di gara, può sovrastimare le gare "bagnate".

**Sensibilità della griglia stimata a eventi anomali**: per le previsioni anticipate (prima delle qualifiche reali), la griglia viene stimata dalle ultime gare del pilota. Una singola griglia anomala può distorcere la stima. Risolto passando da media mobile esponenziale a **mediana** delle ultime 6 griglie, robusta a un singolo outlier, verificato sul caso reale.

**Circuiti debuttanti**: quando un circuito è nuovo in calendario (es. Madring, GP di Spagna 2026, vedi sezione sotto), le feature calcolate dallo storico (velocità media, indice sorpassi) ricadono sulla media globale, non avendo precedenti specifici, previsioni per debutti assoluti vanno lette con più cautela.

**Un singolo DNF può non essere pienamente catturato dalla forma generale**: se le gare precedenti erano forti, la media (anche esponenziale) può non riflettere abbastanza un incidente isolato recente, non ancora corretto con un flag dedicato, possibile miglioramento futuro.

## Gestione di circuiti nuovi in calendario

Quando un circuito debutta (es. Madring/GP Spagna 2026) e non è nel dataset storico Kaggle, va aggiunto manualmente a `data/raw/circuits.csv` e `data/reference/circuit_characteristics.csv` prima di generare previsioni, il sistema non lo fa automaticamente. Coordinate e caratteristiche vanno verificate da fonti affidabili (Wikipedia, sito ufficiale del circuito).

## Struttura del progetto

apex-predictor/
├── api/
│ ├── main.py # API REST (FastAPI)
│ └── static/index.html # interfaccia web
├── data/
│ ├── raw/ # dati grezzi Kaggle + backfill Jolpica (non versionati)
│ └── reference/ # dati curati dal progetto, versionati (circuiti, meteo)
├── docs/ # documentazione di processo (prompt LLM per dati circuiti)
├── models/ # modello + configurazione (non versionati)
├── notebooks/ # EDA, feature engineering, confronto modelli, tuning, analisi finale
├── src/ # codice riutilizzabile e testato
│ ├── data_loading.py, features.py, train.py, predict.py
│ ├── live_predict.py # feature per gare future (no shift necessario)
│ └── jolpica_client.py, weather_client.py
├── scripts/ # script eseguibili standalone
│ ├── run_all.sh # aggiorna dati + riallena modello
│ ├── start_api.sh # avvia il server
│ └── update_and_serve.sh # entrambi in sequenza
├── run_pipeline.py, try_predictions.py
├── requirements.txt
└── README.md


## Stato del progetto

✅ Pipeline completa end-to-end, con analisi rigorosa delle feature e workflow automatizzato in un solo comando.

## Roadmap

- [x] EDA, feature engineering, confronto modelli, tuning
- [x] Backfill storico 2025-2026 (risultati, qualifiche, meteo)
- [x] Feature complete, poi selezionate a 14 dopo analisi SHAP
- [x] Correzione stima griglia (mediana robusta agli outlier)
- [x] Workflow in un comando (`run_all.sh`)
- [x] Interfaccia web (FastAPI + pagina HTML)
- [ ] Bot Telegram (stesso backend, framework diverso)