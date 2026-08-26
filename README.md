# Apex Predictor

Modello di classificazione binaria per prevedere se un pilota di Formula 1 finirà sul podio (top 3) in una gara, basato su dati storici 1950-2024.

## Obiettivo del progetto

L'obiettivo tecnico è costruire una pipeline end-to-end — dalla pulizia dati grezzi multi-tabella fino a un modello di classificazione validato — evitando errori comuni come il data leakage temporale (usare informazioni non disponibili al momento della previsione).

## Dataset

[Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020) di rohanrao su Kaggle, basato sui dati storici Ergast/Jolpica-F1. Struttura relazionale multi-tabella (gare, risultati, piloti, costruttori, qualifiche, standings).

> I dati grezzi non sono inclusi nel repository (vedi `.gitignore`). 
> Per scaricarli:
> 1. Installa il Kaggle CLI: `pip install kaggle` (già in `requirements.txt`)
> 2. Genera un token API da kaggle.com/settings → API → Create New Token
> 3. Posizionalo in `~/.kaggle/kaggle.json` (permessi `chmod 600`)
> 4. Esegui `./scripts/download_data.sh`

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