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

## Modello finale: Random Forest

Confrontati due algoritmi (Logistic Regression, Random Forest) e diverse soglie di decisione, sullo stesso split temporale (train 2004-2022, test 2023-2024). 
Il migliore: **Random Forest, soglia 0.6**.

Risultati sulla classe Podio:
- Precision: 0.54
- Recall: 0.88 (122 podi individuati su 138 reali)
- F1-score: 0.67
- Accuracy complessiva: 0.87

**Feature importance**: la posizione in griglia resta il fattore singolo più predittivo (43%), ma la forma recente del pilota (punti + posizione media) pesa quasi altrettanto in blocco (46%). Storico sul circuito (9%) e affidabilità scuderia (2.5%) contribuiscono meno del previsto.


| Esperimento                            | Precision | Recall   | F1       |
|----------------------------------------|-----------|----------|----------|
| Logistic Regression, soglia 0.5        | 0.42      | 0.91     | 0.58     |
| Logistic Regression, soglia 0.8        | 0.59      | 0.67     | 0.63     |
| Random Forest, soglia 0.5              | 0.49      | 0.91     | 0.64     |
| **Random Forest, soglia 0.6 (finale)** | **0.54**  | **0.88** | **0.67** |

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

## Stato del progetto

🚧 In sviluppo — Fasi 1, 2 e 3 completate: EDA, feature engineering, modello finale selezionato e validato (Random Forest).

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage
- [x] Confronto modelli (Logistic Regression, Random Forest) e selezione finale
- [x] Codice trasferito in `src/` (moduli riutilizzabili e testabili)
- [ ] Pipeline di aggiornamento con dati recenti (API Jolpica-F1)