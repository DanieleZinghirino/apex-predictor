# Apex Predictor

Sistema di machine learning che prevede la probabilità che un pilota di Formula 1 finisca sul podio (top 3), usando dati storici (2004-2026, aggiornati automaticamente da fonte live) e un modello XGBoost ottimizzato.

## Indice
- [Come funziona, in breve](#come-funziona-in-breve)
- [Setup](#setup)
- [Workflow completo](#workflow-completo)
- [Struttura del progetto](#struttura-del-progetto)
- [Metodologia e decisioni chiave](#metodologia-e-decisioni-chiave)
- [Risultati del modello](#risultati-del-modello)
- [Roadmap](#roadmap)

## Come funziona, in breve

1. Dati storici F1 (Kaggle, 2004-2024) + aggiornamento automatico via API (Jolpica-F1, 2025-oggi)
2. Feature engineering senza data leakage temporale (ogni feature usa solo gare precedenti a quella prevista)
3. Modello XGBoost, validato e ottimizzato, calibrato per **precision alta** (il progetto genera previsioni condivise pubblicamente, dove un falso allarme è visibile e costa credibilità)
4. Script che genera la previsione per la prossima gara reale, usando la griglia di qualifica ufficiale (o una stima, se non ancora disponibile)

## Setup

```bash
git clone <url-repo>
cd apex-predictor
python3 -m venv venv
source venv/bin/activate   # su Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow completo

Questi comandi vanno eseguiti **in ordine**, dalla root del progetto, con il virtual environment attivo.

### 1. Scarica i dati storici (Kaggle, 2004-2024)

```bash
python3 -c "print('Genera un token API da kaggle.com/settings → API → Create New Token')"
# posiziona il token in ~/.kaggle/kaggle.json (chmod 600)
./scripts/download_data.sh
```

Scarica ed estrae automaticamente in `data/raw/`.

### 2. Estendi lo storico con le stagioni recenti (Jolpica-F1)

```bash
python3 scripts/update_data.py
```

Scarica tutte le gare disputate dal 2025 a oggi non ancora presenti in locale, le integra nei CSV storici (con mappatura automatica di piloti/costruttori/circuiti nuovi). **Idempotente**: rieseguibile in qualunque momento, salta le gare già presenti. Fa un backup automatico prima di scrivere.

### 3. Allena il modello

```bash
python3 run_pipeline.py
```

Carica i dati, costruisce le feature (piloti, scuderie, circuiti), allena XGBoost, trova la soglia di decisione ottimale sul test set più recente, valuta, e salva modello + configurazione in `models/`.

### 4. Genera la previsione per la prossima gara

```bash
python3 scripts/predict_next_race.py
```

Recupera automaticamente la prossima gara in calendario. Se la griglia di qualifica ufficiale è già disponibile, genera una previsione **DEFINITIVA**; altrimenti genera una previsione **ANTICIPATA**, con griglia stimata dalla forma recente dei piloti — sempre etichettata chiaramente come tale.

### (Opzionale) Prova il modello su una gara già disputata

```bash
python3 try_predictions.py
```

Utile per verificare rapidamente il comportamento del modello senza aspettare una gara futura — confronta la previsione con il risultato reale già noto.

## Struttura del progetto

apex-predictor/
├── data/
│ ├── raw/ # dati grezzi Kaggle + backfill Jolpica (non versionati)
│ ├── processed/ # dati puliti intermedi (non versionati)
│ └── reference/ # dati curati dal progetto, VERSIONATI (es. caratteristiche circuiti)
├── docs/ # documentazione di processo (es. prompt usati per dati esterni)
├── notebooks/ # notebook di esplorazione: EDA, feature engineering, confronto modelli, tuning
├── src/ # codice riutilizzabile e testato
│ ├── data_loading.py # caricamento CSV, costruzione dataset di lavoro
│ ├── features.py # feature engineering (piloti, scuderie, circuiti), no leakage
│ ├── train.py # training, ricerca soglia, valutazione, salvataggio modello
│ ├── predict.py # caricamento modello salvato, previsioni su nuovi dati
│ ├── live_predict.py # costruzione feature per gare FUTURE (no shift necessario)
│ └── jolpica_client.py # client per l'API Jolpica-F1
├── scripts/ # script eseguibili standalone
│ ├── download_data.sh # scarica il dataset storico da Kaggle
│ ├── update_data.py # estende lo storico con le stagioni recenti
│ └── predict_next_race.py # genera la previsione per la prossima gara
├── models/ # modello addestrato + configurazione (non versionati)
├── run_pipeline.py # pipeline di training end-to-end
├── try_predictions.py # demo: previsione su gara già disputata
├── requirements.txt
└── README.md


## Metodologia e decisioni chiave

**Range dati storici: 2004-2024** (poi esteso a 2025-2026 via API). Il tracciamento del giro veloce, introdotto nel 2004, si è rivelato fortemente predittivo del podio — motivo per cui il periodo precedente (2000-2003) è stato escluso a favore di un dataset completo su tutte le feature.

**Target basato su `positionOrder`, non `position`**, perché quest'ultima è NaN per i ritiri, perdendo proprio i casi più utili da classificare come "non podio".

**Feature engineering senza data leakage**: ogni feature (forma pilota, affidabilità scuderia, storico circuito) usa esclusivamente gare precedenti a quella prevista — verificato manualmente su singoli piloti prima di essere esteso a tutto il dataset. Per le gare future (dove non serve alcuno shift, dato che la gara non è nello storico), `src/live_predict.py` replica la stessa logica in una forma adattata.

**Selezione del modello**: confrontati 8 algoritmi di classificazione, poi ottimizzati i due migliori (Random Forest, XGBoost) con `RandomizedSearchCV` + `TimeSeriesSplit` (validazione che rispetta l'ordine cronologico). XGBoost tunato è il modello finale, scelto per **precision alta** — il progetto genera previsioni pubbliche, dove un falso allarme è visibile e costa credibilità più di un podio mancato.

**Feature sui circuiti**: caratteristiche fisiche (lunghezza, curve, direzione, altitudine, carico aerodinamico) da una tabella compilata con l'aiuto di un LLM, ridotta alle sole colonne con buona coerenza tra fonti multiple — le colonne più variabili sono state sostituite con equivalenti calcolati direttamente dallo storico gare (velocità media, indice di sorpassabilità), dati reali invece di stime esterne. Dettaglio completo in `docs/circuit_data_prompt.md`.

Dettaglio completo del ragionamento in `notebooks/01_eda.ipynb` → `04_hyperparameter_tuning.ipynb`, in ordine.

## Risultati del modello

Validato su gare mai viste in training (2025 e parte del 2026):

| Metrica (classe Podio) | Valore |
|---|---|
| Precision | 0.650 |
| Recall | 0.824 |
| F1-score | 0.727 |
| Accuracy complessiva | 0.91 |

**Feature più importanti**: posizione in griglia (43%) e forma recente del pilota (28% in blocco) dominano; le feature sui circuiti (statiche + calcolate) contribuiscono complessivamente circa il 13%, non marginale ma neanche decisivo.

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage (piloti, scuderie, circuiti)
- [x] Confronto sistematico tra 8 algoritmi di classificazione
- [x] Hyperparameter tuning con validazione temporale
- [x] Backfill storico 2025-2026 via Jolpica-F1
- [x] Pipeline di previsione sulla prossima gara (con fallback su griglia stimata)
- [ ] API/demo per rendere le previsioni consultabili senza terminale