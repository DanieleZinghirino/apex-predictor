# Apex Predictor

Modello di classificazione binaria per prevedere se un pilota di Formula 1 finirà sul podio (top 3), basato su dati storici 2004-2024.

## Obiettivo del progetto

L'obiettivo tecnico è costruire una pipeline end-to-end, dalla pulizia dati multi-tabella fino a un modello di classificazione validato, evitando errori comuni come il data leakage temporale.

## Dataset

[Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) di rohanrao su Kaggle, basato sui dati Ergast. Struttura relazionale multi-tabella (gare, risultati, piloti, costruttori, qualifiche).

I dati grezzi non sono inclusi nel repository. Per scaricarli:
1. `pip install -r requirements.txt` (include il Kaggle CLI)
2. Genera un token API da kaggle.com/settings → API → Create New Token
3. Posizionalo in `~/.kaggle/kaggle.json` (`chmod 600`)
4. Esegui `./scripts/download_data.sh`

## Cosa dice l'analisi esplorativa

- **Classi sbilanciate**: solo il 14.3% dei risultati è un podio. Useremo precision, recall, F1-score e matrice di confusione, non solo accuracy.
- **La griglia conta, ma non decide tutto**: correlazione griglia↔posizione finale = 0.579 (moderata-forte), c'è margine per altre feature.
- **Il giro veloce è un forte segnale di podio**: rank medio 3.6 per chi fa podio contro 11.5 per chi non lo fa.

## Feature engineering

Ogni feature è calcolata usando **solo gare precedenti** a quella in esame, per evitare data leakage temporale; il modello non deve mai "vedere" informazioni che non avrebbe nella realtà al momento della previsione.

- **Forma recente del pilota**: media punti e posizione finale sulle ultime 10 gare precedenti;
- **Affidabilità della scuderia**: percentuale di gare completate (non ritirate) sulle ultime 10 gare del costruttore;
- **Storico pilota-circuito**: media posizione del pilota su quello specifico circuito, su tutte le apparizioni precedenti (non una finestra fissa: le apparizioni per circuito sono naturalmente rare, una all'anno).

**Gestione dei valori mancanti**: le prime gare in assoluto di un pilota/scuderia nel dataset non hanno storico da cui calcolare nulla: quelle righe (~1-2% del totale) vengono eliminate. Per lo storico pilota-circuito, mancante nel 26% dei casi (molto più comune: basta non aver mai corso su quel tracciato specifico), si usa la forma generale del pilota come stima di fallback.

## Modello baseline

Logistic Regression, `class_weight="balanced"`, split temporale (train 2004-2022, test 2023-2024).

Risultati sulla classe Podio:
- Precision: 0.42
- Recall: 0.91
- F1-score: 0.58
- Accuracy complessiva: 0.80

Il modello preferisce nettamente il falso allarme alla previsione mancata (126 podi individuati su 138 reali, ma 174 falsi positivi su 300 podi previsti), conseguenza diretta di `class_weight="balanced"`. Prossimo step: confronto con soglie di decisione diverse e Random Forest.


## Decisioni chiave

- **Range dati: 2004-2024**, non l'intero storico. Il tracciamento del giro veloce (introdotto nel 2004) è risultato troppo predittivo per scartarlo restando su un range più ampio.
- **Target basato su `positionOrder`, non su `position`**, perché quest'ultima è NaN per i ritiri, perdendo proprio i casi più utili da classificare come "non podio".

Dettaglio completo del ragionamento nei notebook `01_eda.ipynb` e `02_feature_engineering.ipynb`.

## Modello finale: Random Forest

Confrontati due algoritmi (Logistic Regression, Random Forest) e diverse soglie di decisione, sempre sullo stesso split temporale (train 2004-2022, test 2023-2024). Il migliore: **Random Forest, soglia 0.6**.

Risultati sulla classe Podio:
- Precision: 0.54
- Recall: 0.88 (122 podi individuati su 138 reali)
- F1-score: 0.67
- Accuracy complessiva: 0.87

**Feature importance**: la posizione in griglia resta il singolo fattore più predittivo (43%), ma la forma recente del pilota (punti + posizione media) pesa quasi altrettanto in blocco (46%). Storico sul circuito (9%) e affidabilità scuderia (2.5%) contribuiscono meno del previsto.

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

## Stato del progetto

🚧 In sviluppo — Fasi 1, 2 e 3 completate: EDA, feature engineering, modello finale selezionato e validato (Random Forest).

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage
- [x] Confronto modelli (Logistic Regression, Random Forest) e selezione finale
- [ ] Pipeline di aggiornamento con dati recenti (API Jolpica-F1)