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

## Analisi esplorativa — cosa è emerso

- Target sbilanciato: solo il 14.3% dei risultati è un podio → accuratezza fuorviante, priorità a precision/recall/F1
- Griglia di partenza correlata alla posizione finale (0.579) ma non decisiva
- Giro veloce fortemente predittivo del podio (rank medio 3.6 vs 11.5) → ha motivato il restringimento del dataset al 2004-2024, unico periodo in cui questo dato è tracciato

## Feature engineering

Ogni feature usa solo gare precedenti a quella in esame (mai la gara corrente o future), verificato manualmente prima di essere esteso a tutto il dataset:

- **Forma recente pilota** — media punti/posizione ultime 10 gare
- **Affidabilità scuderia** — % gare completate ultime 10 uscite del costruttore
- **Storico pilota-circuito** — media posizione su quel tracciato, tutte le apparizioni precedenti

Valori mancanti: eliminati per forma pilota/affidabilità scuderia (~1-2%, prime apparizioni in assoluto); per lo storico circuito (26% mancante) si usa la forma generale del pilota come fallback, con un flag esplicito (`no_circuit_history`) che segnala al modello quando è una stima.

## Dal confronto al modello campione

Dopo il primo modello (Random Forest, F1 0.674), sono stati condotti due approfondimenti per validare e rifinire la scelta.

**Confronto sistematico** — 8 algoritmi testati sullo stesso split temporale (Decision Tree, Logistic Regression, KNN, Naive Bayes, Random Forest, Gradient Boosting, XGBoost, SVM). Random Forest si è confermato il migliore, ma con margine ridotto su Decision Tree — segnale che un albero singolo ben regolarizzato cattura già gran parte del segnale disponibile con queste 6 feature.

**Hyperparameter tuning** — Random Forest e XGBoost (il modello con più margine di miglioramento nel confronto) sono stati ottimizzati con `RandomizedSearchCV` e `TimeSeriesSplit` (validazione incrociata che rispetta l'ordine cronologico, mai mescolando futuro e passato). Il tuning ha ribaltato il verdetto: **XGBoost tunato è il nuovo modello campione**.

| Modello | Soglia | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest (originale) | 0.60 | 0.545 | 0.884 | 0.674 |
| Random Forest (tunato) | 0.80 | 0.664 | 0.674 | 0.669 |
| XGBoost (originale) | 0.40 | 0.509 | 0.855 | 0.638 |
| **XGBoost (tunato, finale)** | **0.75** | **0.628** | **0.746** | **0.682** |

**Perché XGBoost tunato, nonostante il margine di F1 minimo**: la scelta finale non si basa solo sul punteggio aggregato, ma sul caso d'uso del progetto — generare previsioni condivise pubblicamente con persone interessate. In questo contesto un falso allarme è visibile e verificabile (la gara si corre, l'errore si vede), quindi la **precision alta** (0.628, la più alta tra tutti i modelli testati) conta più della recall: meglio segnalare meno podi ma con più affidabilità, piuttosto che coprirne di più a costo di previsioni sbagliate frequenti.

Dettaglio completo in `notebooks/03_model_comparison.ipynb` e `notebooks/04_hyperparameter_tuning.ipynb`.

## Feature importance (XGBoost, modello finale)

- **`grid`** (49%) — la posizione di partenza pesa ancora di più che in Random Forest (43%): XGBoost, ottimizzato per precision alta, si affida fortemente al segnale più diretto e affidabile disponibile
- **`driver_recent_position_avg`** (32%) — la forma recente del pilota resta il secondo fattore più importante, sostanzialmente in linea con Random Forest
- **`driver_recent_points_avg`** (9%) — contributo più contenuto rispetto a Random Forest, probabile ridondanza parziale con la feature precedente (le due sono correlate, e XGBoost tende a "specializzarsi" su una delle due varianti ridondanti più di quanto faccia Random Forest)
- **`no_circuit_history`, `driver_circuit_avg_position`, `constructor_reliability`** (4%, 3%, 2%) — contributo marginale, coerente con quanto già osservato in precedenza

Il pattern generale conferma quanto visto con Random Forest: griglia e forma recente del pilota dominano nettamente, mentre storico circuito e affidabilità scuderia aggiungono poco. XGBoost tunato si affida ancora più fortemente alla griglia (49% vs 43%) — coerente con la sua calibrazione verso la precision: il segnale più diretto e meno rumoroso riduce il rischio di falsi allarmi.

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

Random Forest risultava il migliore, ma con margine ridotto su Decision Tree — segnale che un singolo albero ben regolarizzato cattura già gran parte del segnale disponibile con queste 6 feature. XGBoost, sorprendentemente indietro rispetto alla sua reputazione su dati tabellari, non aveva ricevuto alcun tuning in questo confronto — un'ipotesi poi confermata (vedi sezione successiva). Dettaglio in `notebooks/03_model_comparison.ipynb`.

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

Carica i dati, costruisce le feature, allena il modello Random Forest, lo valuta sul test set 2023-2024 e salva modello + configurazione in `models/`.
I percorsi dei dati e dei modelli sono calcolati rispetto alla root del progetto (non alla working directory corrente), quindi lo script funziona correttamente da qualunque cartella lo si lanci.

## Stato del progetto

🚧 In sviluppo — Fasi 1, 2 e 3 completate: EDA, feature engineering, modello finale selezionato e validato (Random Forest).

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage
- [x] Confronto modelli e selezione finale
- [x] Codice trasferito in `src/` (moduli riutilizzabili e testabili)
- [ ] Pipeline di aggiornamento con dati recenti (API Jolpica-F1)