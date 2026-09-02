# Apex Predictor

Modello di classificazione binaria per prevedere se un pilota di Formula 1 finirà sul podio (top 3), basato su dati storici 2004-2024.

## Dataset

[Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) di rohanrao su Kaggle, struttura multi-tabella (gare, risultati, piloti, costruttori, qualifiche).

Dati non inclusi nel repository. Per scaricarli:
```bash
pip install -r requirements.txt
# token da kaggle.com/settings → API → Create New Token
# posizionalo in ~/.kaggle/kaggle.json (chmod 600)
./scripts/download_data.sh
```

## Analisi esplorativa, cosa è emerso

- Target sbilanciato: solo il 14.3% dei risultati è un podio → accuratezza fuorviante, priorità a precision/recall/F1
- Griglia di partenza correlata alla posizione finale (0.579) ma non decisiva
- Giro veloce fortemente predittivo del podio (rank medio 3.6 vs 11.5) → ha motivato il restringimento del dataset al 2004-2024, unico periodo in cui questo dato è tracciato

## Feature engineering

Ogni feature usa solo gare precedenti a quella in esame (mai la gara corrente o future), verificato manualmente prima di essere esteso a tutto il dataset:

- **Forma recente pilota**, media punti/posizione ultime 10 gare
- **Affidabilità scuderia**, % gare completate ultime 10 uscite del costruttore
- **Storico pilota-circuito**, media posizione su quel tracciato, tutte le apparizioni precedenti

Valori mancanti: eliminati per forma pilota/affidabilità scuderia (~1-2%, prime apparizioni in assoluto); per lo storico circuito (26% mancante) si usa la forma generale del pilota come fallback, con un flag esplicito (`no_circuit_history`) che segnala al modello quando è una stima.

## Confronto sistematico tra modelli

Come primo passo di validazione, sono stati confrontati 8 algoritmi (Decision Tree, Logistic Regression, KNN, Naive Bayes, Random Forest, Gradient Boosting, XGBoost, SVM), tutti sullo stesso split temporale, ciascuno alla propria soglia di decisione ottimale (massimo F1), **senza hyperparameter tuning**.

| Modello | F1 (classe Podio) |
|---|---|
| Random Forest | 0.674 |
| Decision Tree | 0.664 |
| KNN | 0.657 |
| Gradient Boosting | 0.655 |
| Logistic Regression | 0.641 |
| XGBoost | 0.638 |
| SVM | 0.625 |
| Naive Bayes | 0.598 |

Random Forest risultava il migliore, ma con margine ridotto su Decision Tree, segnale che un singolo albero ben regolarizzato cattura già gran parte del segnale disponibile con queste 6 feature. XGBoost, sorprendentemente indietro rispetto alla sua reputazione su dati tabellari, non aveva ricevuto alcun tuning in questo confronto, un'ipotesi poi confermata nella sezione successiva. Dettaglio in `notebooks/03_model_comparison.ipynb`.

## Modello campione finale

Il modello (XGBoost, iperparametri da `RandomizedSearchCV` + `TimeSeriesSplit`, dettaglio in `notebooks/04_hyperparameter_tuning.ipynb`) è addestrato su dati fino al 2024 e validato sulle stagioni 2025-2026, integrate nello storico tramite l'API Jolpica-F1 (vedi sezione successiva).

| | Valore |
|---|---|
| Soglia di decisione | 0.75 |
| Precision (Podio) | 0.629 |
| Recall (Podio) | 0.833 |
| F1 (Podio) | 0.717 |
| Accuracy complessiva | 0.90 |

La soglia è calibrata per **precision alta**: il progetto genera previsioni condivise pubblicamente, dove un falso allarme è visibile e verificabile appena la gara si corre, meglio segnalare meno podi con più affidabilità.

Nota: questi numeri sono leggermente migliori di quelli misurati sul test set precedente (2023-2024, F1 0.682), segnale incoraggiante di buona generalizzazione su dati più recenti, non ancora un pattern consolidato su cui trarre conclusioni definitive.

Dettaglio completo in `notebooks/04_hyperparameter_tuning.ipynb`.

## Feature importance (XGBoost, modello finale)

- **`grid`** (49%), la posizione di partenza pesa ancora di più che in Random Forest (43%): XGBoost, ottimizzato per precision alta, si affida fortemente al segnale più diretto e affidabile disponibile
- **`driver_recent_position_avg`** (32%), la forma recente del pilota resta il secondo fattore più importante, sostanzialmente in linea con Random Forest
- **`driver_recent_points_avg`** (9%), contributo più contenuto rispetto a Random Forest, probabile ridondanza parziale con la feature precedente
- **`no_circuit_history`, `driver_circuit_avg_position`, `constructor_reliability`** (4%, 3%, 2%), contributo marginale, coerente con quanto già osservato in precedenza

Il pattern generale conferma quanto visto con Random Forest: griglia e forma recente del pilota dominano nettamente, mentre storico circuito e affidabilità scuderia aggiungono poco. XGBoost tunato si affida ancora più fortemente alla griglia, coerente con la sua calibrazione verso la precision: il segnale più diretto e meno rumoroso riduce il rischio di falsi allarmi.

## Aggiornamento dati live (Jolpica-F1)

Il dataset storico Kaggle si fermava al 2024. `src/jolpica_client.py` fornisce un client per l'API [Jolpica-F1](https://github.com/jolpica/jolpica-f1), usato da `scripts/update_data.py` per integrare le stagioni 2025 e parte del 2026 (fino al 23 agosto 2026) nello storico locale, con mappatura automatica tra gli ID testuali di Jolpica (`driverRef`, `constructorRef`, `circuitRef`) e gli ID numerici del dataset Kaggle, creando nuovi ID per piloti/costruttori non presenti nel dump originale

Per rieseguire l'aggiornamento (idempotente, salta le gare già presenti):
```bash
python3 scripts/update_data.py
```

## Struttura del progetto

apex-predictor/
├── data/
│ ├── raw/ # dati grezzi (non versionati)
│ └── processed/ # dati puliti pronti per il training
├── notebooks/ # notebook di esplorazione e sviluppo
├── src/ # codice riutilizzabile
├── scripts/ # script di utilità
├── models/ # modelli addestrati salvati (non versionati)
├── requirements.txt
└── README.md

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Eseguire la pipeline completa

Oltre ai notebook (che documentano il processo esplorativo), il progetto include una pipeline eseguibile end-to-end:

```bash
python3 run_pipeline.py
```

Carica i dati, costruisce le feature, allena il modello campione (XGBoost tunato), lo valuta sul test set 2023-2024 e salva modello + configurazione in `models/`. I percorsi dei dati e dei modelli sono calcolati rispetto alla root del progetto, quindi lo script funziona correttamente da qualunque cartella lo si lanci.

## Stato del progetto

🚧 In sviluppo — Storico esteso a 2025-2026 via Jolpica-F1, modello ri-addestrato e validato sul periodo più recente. Prossimo passo: generazione previsioni sulla prossima gara.

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage
- [x] Confronto sistematico tra 8 algoritmi di classificazione
- [x] Hyperparameter tuning con validazione temporale
- [x] Codice trasferito in `src/` (moduli riutilizzabili e testabili)
- [x] Backfill storico 2025-2026 via Jolpica-F1
- [ ] Generazione previsioni sulla prossima gara (Gran Premio d'Italia, 6 settembre 2026)