# Prompt per la raccolta dati sui circuiti

Usato per generare `data/reference/circuit_characteristics.csv` tramite un LLM esterno. Include l'istruzione esplicita di dichiarare i dati non verificabili invece di ometterli (vedi nota sotto).

## Prompt originale

Ho bisogno di una tabella dati sui circuiti di Formula 1, per un progetto di machine learning che prevede i risultati delle gare. Mi servono caratteristiche tecniche verificabili, non stime approssimative.

ELENCO CIRCUITI (uno per riga, formato circuitRef | nome | paese):
[elenco ottenuto da data/raw/circuits.csv, filtrato sui circuiti usati dal 2004 in poi]

Per CIASCUN circuito della lista, forniscimi questi dati in formato tabella CSV...
[vedi testo completo del prompt originale]

## Nota metodologica

Prima iterazione: il modello ha lasciato "N/D" sui valori non verificabili con buona fonte. Successivamente è stato chiesto esplicitamente di includere anche stime plausibili ma non verificate ("metti anche dati non attendibili ma almeno plausibili"), il che ha prodotto una seconda tabella completa ma con alcune incongruenze rispetto alla prima (valori diversi per gli stessi circuiti tra le due risposte).

**Decisione presa**: nel dataset finale sono state mantenute solo le colonne con buona coerenza tra le due risposte (length_km, num_corners, direction, altitude_m, downforce_level). Le colonne più variabili (num_straights, longest_straight_m, avg_speed_kmh, track_width) sono state scartate e sostituite, dove possibile, con equivalenti calcolati direttamente dallo storico gare (vedi src/features.py — circuit_avg_speed_history, circuit_overtaking_index), che usano dati reali invece di stime.  