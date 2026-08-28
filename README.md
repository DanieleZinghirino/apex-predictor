# Apex Predictor

Modello di classificazione binaria per prevedere se un pilota di Formula 1 finirà sul podio (top 3), basato su dati storici 2004-2024.

## Obiettivo del progetto

L'obiettivo tecnico è costruire una pipeline end-to-end — dalla pulizia dati multi-tabella fino a un modello di classificazione validato — evitando errori comuni come il data leakage temporale.

## Dataset

[Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) di rohanrao su Kaggle, basato sui dati Ergast. Struttura relazionale multi-tabella (gare, risultati, piloti, costruttori, qualifiche).

I dati grezzi non sono inclusi nel repository. Per scaricarli:
1. `pip install -r requirements.txt` (include il Kaggle CLI)
2. Genera un token API da kaggle.com/settings → API → Create New Token
3. Posizionalo in `~/.kaggle/kaggle.json` (`chmod 600`)
4. Esegui `./scripts/download_data.sh`

## Cosa dice l'analisi esplorativa

- **Classi sbilanciate**: solo il 14.3% dei risultati è un podio. Useremo precision, recall, F1-score e matrice di confusione, non solo accuracy.
- **La griglia conta, ma non decide tutto**: correlazione griglia↔posizione finale = 0.579 (moderata-forte) — c'è margine per altre feature.
- **Il giro veloce è un forte segnale di podio**: rank medio 3.6 per chi fa podio contro 11.5 per chi non lo fa.

## Feature engineering

Ogni feature è calcolata usando **solo gare precedenti** a quella in esame (`shift()` + `rolling()`/`expanding()` di pandas), per evitare data leakage temporale — il modello non deve mai "vedere" informazioni che non avrebbe nella realtà al momento della previsione.

- **Forma recente del pilota**: media punti e posizione finale sulle ultime 10 gare precedenti
- **Affidabilità della scuderia**: percentuale di gare completate (non ritirate) sulle ultime 10 gare del costruttore — finestra più ampia della forma pilota, perché l'affidabilità meccanica cambia più lentamente
- **Storico pilota-circuito**: media posizione del pilota su quello specifico circuito, su tutte le apparizioni precedenti (non una finestra fissa: le apparizioni per circuito sono naturalmente rare, una all'anno)

**Gestione dei valori mancanti**: le prime gare in assoluto di un pilota/scuderia nel dataset non hanno storico da cui calcolare nulla — quelle righe (~1-2% del totale) vengono eliminate. Per lo storico pilota-circuito, mancante nel 26% dei casi (molto più comune: basta non aver mai corso su quel tracciato specifico), si usa la forma generale del pilota come stima di fallback, accompagnata da un flag esplicito (`no_circuit_history`) che segnala al modello quando manca lo storico reale.

## Modello baseline

Logistic Regression, `class_weight="balanced"`, split temporale (train 2004-2022, test 2023-2024).

Risultati sulla classe Podio:
- Precision: 0.42
- Recall: 0.91
- F1-score: 0.58
- Accuracy complessiva: 0.80

Il modello preferisce nettamente il falso allarme alla previsione mancata (126 podi individuati su 138 reali, ma 174 falsi positivi su 300 podi previsti) — conseguenza diretta di `class_weight="balanced"`. Prossimo step: confronto con soglie di decisione diverse e Random Forest.


## Decisioni chiave

- **Range dati: 2004-2024**, non l'intero storico. Il tracciamento del giro veloce (introdotto nel 2004) è risultato troppo predittivo per scartarlo restando su un range più ampio.
- **Target basato su `positionOrder`, non su `position`**, perché quest'ultima è NaN per i ritiri, perdendo proprio i casi più utili da classificare come "non podio".

Dettaglio completo del ragionamento nei notebook `01_eda.ipynb` e `02_feature_engineering.ipynb`.

## Struttura del progetto

## Struttura del progetto
apex-predictor/
├── data/
│ ├── raw/ # dati grezzi (non versionati)
│ └── processed/ # dati puliti pronti per il training
├── notebooks/ # notebook di esplorazione e sviluppo
├── src/ # codice riutilizzabile (preprocessing, training, evaluation)
├── requirements.txt
└── README.md


## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Stato del progetto

🚧 In sviluppo — Fase 1 e 2 completate. Fase 3: baseline addestrato e valutato, confronto con soglie alternative e Random Forest in corso.

## Roadmap

- [x] Setup ambiente e struttura progetto
- [x] Analisi esplorativa dei dati
- [x] Feature engineering senza data leakage
- [x] Modello baseline (Logistic Regression)
- [ ] Confronto con Random Forest e soglie di decisione alternative
- [ ] Pipeline di aggiornamento con dati recenti (API Jolpica-F1)