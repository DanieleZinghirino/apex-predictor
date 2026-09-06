# Apex Predictor 🏎️

Sistema di machine learning che prevede la probabilità che un pilota di Formula 1 finisca sul podio (top 3), usando dati storici (2004-2026, aggiornati automaticamente da fonte live) e un modello XGBoost ottimizzato, con 22 feature su piloti, scuderie, circuiti, qualifica e meteo.

## Come funziona, in breve

1. Dati storici F1 (Kaggle, 2004-2024) + aggiornamento automatico via API (Jolpica-F1, 2025-oggi)
2. Feature engineering senza data leakage temporale
3. Modello XGBoost, calibrato per **precision alta** (previsioni condivise pubblicamente, dove un falso allarme è visibile e costa credibilità)
4. Previsione sulla prossima gara reale, con griglia e meteo effettivi quando disponibili, stimati altrimenti

## Setup

```bash
git clone <url-repo>
cd apex-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow completo

```bash
# 1. Dati storici Kaggle (richiede token in ~/.kaggle/kaggle.json)
./scripts/download_data.sh

# 2. Estendi con le stagioni recenti (risultati + qualifiche)
python3 scripts/update_data.py

# 3. Meteo storico (una tantum, o dopo un update_data.py)
python3 scripts/fetch_weather.py

# 4. Allena il modello
python3 run_pipeline.py

# 5. Previsione per la prossima gara (griglia/meteo reali se disponibili)
python3 scripts/predict_next_race.py

# (Opzionale) Demo su una gara già disputata
python3 try_predictions.py
```

## Feature del modello (22 totali)

**Piloti/scuderie**: forma recente (punti, posizione), affidabilità scuderia, storico su circuito, confronto col compagno di squadra, posizione in classifica generale, gara di casa.

**Circuito**: lunghezza, curve, altitudine, carico aerodinamico (da tabella compilata con LLM, verificata su fonti multiple — dettaglio in `docs/circuit_data_prompt.md`), velocità media storica e indice di sorpassabilità (calcolati dai dati reali, non stimati).

**Qualifica**: distacco dal poleman in secondi.

**Meteo**: temperatura massima, precipitazioni (Open-Meteo — archivio storico per il training, previsione per le gare future).

## Risultati del modello

Validato su gare mai viste in training (2025 e parte del 2026):

| Metrica (classe Podio) | Valore |
|---|---|
| Precision | 0.624 |
| Recall | 0.861 |
| F1-score | 0.724 |
| Accuracy | 0.90 |
| Soglia | 0.65 |

### Evoluzione del modello

| Fase | Precision | Recall | F1 |
|---|---|---|---|
| XGBoost tunato, test 2023-2024 | 0.628 | 0.746 | 0.682 |
| + backfill storico 2025-2026 | 0.629 | 0.833 | 0.717 |
| + feature circuiti | 0.650 | 0.824 | 0.727 |
| + gara di casa, standings, compagno, qualifica, meteo | 0.624 | 0.861 | 0.724 |

L'ultimo set di feature non migliora F1 in modo netto, ma sposta il modello verso più recall — segnale che alcune delle nuove feature (es. qualifica, meteo) aggiungono varietà al segnale più che pura precisione. Nessun cambiamento drammatico, coerente con un modello ormai vicino a un plateau con questo tipo di dati tabellari.

## Metodologia e decisioni chiave

**Range dati: 2004-2024**, poi esteso a 2025-2026 via Jolpica-F1. Il tracciamento del giro veloce (dal 2004) è risultato fortemente predittivo, motivando questa scelta.

**Target su `positionOrder`**, non `position` (NaN per i ritiri).

**Nessun data leakage temporale**: ogni feature dinamica usa solo dati precedenti alla gara prevista — verificato manualmente su casi singoli prima di essere esteso a tutto il dataset. Standings: presi dalla gara precedente nello stesso anno (mai dalla gara corrente, che includerebbe già il suo risultato).

**Modello**: confrontati 8 algoritmi, ottimizzati i due migliori con `RandomizedSearchCV` + `TimeSeriesSplit`. XGBoost tunato è il campione, scelto per precision alta.

**Feature sui circuiti**: raccolte con un LLM, con istruzione esplicita di dichiarare i dati non verificabili. Le colonne meno coerenti tra risposte multiple sono state scartate a favore di equivalenti calcolati dai dati reali.

Dettaglio completo in `notebooks/01_eda.ipynb` → `04_hyperparameter_tuning.ipynb`.

## Struttura del progetto

apex-predictor/
├── data/
│ ├── raw/ # dati grezzi Kaggle + backfill Jolpica (non versionati)
│ └── reference/ # dati curati dal progetto, versionati (circuiti, meteo)
├── docs/ # documentazione di processo
├── notebooks/ # EDA, feature engineering, confronto modelli, tuning
├── src/ # codice riutilizzabile e testato
│ ├── data_loading.py, features.py, train.py, predict.py
│ ├── live_predict.py # feature per gare future (no shift necessario)
│ └── jolpica_client.py, weather_client.py
├── scripts/ # script eseguibili standalone
├── models/ # modello + configurazione (non versionati)
├── run_pipeline.py, try_predictions.py
├── requirements.txt
└── README.md


## Stato del progetto

✅ Pipeline completa end-to-end, funzionante e testata: training, backfill storico, previsione live (griglia e meteo reali quando disponibili).

## Roadmap

- [x] EDA, feature engineering, confronto modelli, tuning
- [x] Backfill storico 2025-2026 (risultati, qualifiche, meteo)
- [x] Feature complete: circuiti, gara di casa, standings, compagno di squadra, qualifica, meteo
- [x] Previsione live con dati reali (non più fallback fissi)
- [ ] Tappa 3: servire il modello via API + demo