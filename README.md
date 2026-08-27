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

- **Classi sbilanciate**: solo il 14.3% dei risultati è un podio. L'accuratezza da sola sarà una metrica fuorviante — un modello che prevede sempre "non podio" avrebbe già un'accuratezza alta ma inutile. Useremo precision, recall, F1-score e matrice di confusione.
- **La griglia di partenza conta, ma non decide tutto**: correlazione tra posizione di partenza e posizione finale = 0.579. C'è margine reale perché altre variabili — forma recente, affidabilità della scuderia, passo gara — spieghino la parte che la sola griglia non cattura.
- **Il giro veloce è un forte segnale di podio**: rank medio 3.6 per chi fa podio contro 11.5 per chi non lo fa.

## Decisioni chiave

- **Range dati: 2004-2024**, non l'intero storico. Il tracciamento del giro veloce (introdotto nel 2004) è risultato troppo predittivo per scartarlo restando su un range più ampio.
- **Target basato su `positionOrder`, non su `position`**, perché quest'ultima è NaN per i ritiri, perdendo proprio i casi più utili da classificare come "non podio".

Dettaglio completo del ragionamento in `notebooks/01_eda.ipynb`.

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

🚧 In sviluppo — Fase 1: analisi esplorativa e primo modello sui dati storici.

## Roadmap

- [x] Setup ambiente e struttura progetto
- [ ] Analisi esplorativa dei dati (EDA)
- [ ] Feature engineering (senza data leakage temporale)
- [ ] Training e valutazione di modelli di classificazione
- [ ] Pipeline di aggiornamento con dati recenti (API Jolpica-F1)