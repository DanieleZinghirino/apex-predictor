"""
Infrastruttura condivisa per gli esperimenti di deep learning. Centralizza caricamento dati, codifica delle variabili categoriche (driver/costruttore/circuito) per gli embedding,
e la logica di valutazione
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score

from src.train import FEATURE_COL, temporal_split

# Colonne categoriche che le reti useranno come embedding, in aggiunta alle 14 feature numeriche di FEATURE_COL, un'informazione che
# XGBoost non riceve mai in questa forma (l'identità del pilota/scuderia/circuito, non solo le sue statistiche riassuntive)
CAT_COLS = ["driverId", "constructorId", "circuitId"]


def build_categorical_vocab(train_df, cat_col):
    """
    Costruisce il vocabolario (mappa valore -> indice intero) per una colonna categorica, usando solo il train set e mai il test, per
    evitare che il modello veda categorie del futuro durante la codifica
    L'indice 0 è riservato a "sconosciuto" — qualsiasi valore visto nel test set ma mai nel train

    Parametri:
        train_df: DataFrame di training
        cat_col: nome della colonna categorica (es. "driverId")

    Ritorna:
        dict {valore_originale: indice}, con indice 0 riservato a "sconosciuto"
    """
    unique_values = sorted(train_df[cat_col].unique())
    # Si parte da 1: l'indice 0 è riservato, non assegnato a nessun valore reale
    vocab = {val: idx + 1 for idx, val in enumerate(unique_values)}
    return vocab


def encode_categorical(df, cat_col, vocab):
    """
    Applica un vocabolario già costruito a una colonna, mappando i valori non visti in training sull'indice 0 ("sconosciuto") invece
    di sollevare un errore.
    """
    return df[cat_col].map(vocab).fillna(0).astype(int).values


class TabularDataset(Dataset):
    """
    Dataset PyTorch per dati tabellari misti (feature numeriche + categoriche). Compatibile con DataLoader per il training a batch.
    """
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def prepare_dl_data(df, test_start_year=2025):
    """
    Punto di ingresso unico: dal DataFrame con feature già costruite, prepara tutto il necessario per allenare una rete neurale:
    split temporale, normalizzazione delle feature numeriche, codifica delle categoriche, vocabolari

    Parametri:
        df: DataFrame con FEATURE_COL, CAT_COLS, 'podium' già presenti
        test_start_year: stesso parametro di temporal_split in src/train.py

    Ritorna:
        dict con train_dataset, test_dataset, cat_vocabs, num_mean/num_std
    """
    train_df, test_df = temporal_split(df, test_start_year=test_start_year)

    # Normalizzazione delle feature numeriche: le reti neurali sono sensibili alla scala delle feature: 
    # calcoliamo media/std solo sul train, poi le riusiamo identiche sul test
    num_mean = train_df[FEATURE_COL].mean()
    num_std = train_df[FEATURE_COL].std().replace(0, 1)  # evita divisione per zero su feature costanti

    X_train_num = ((train_df[FEATURE_COL] - num_mean) / num_std).values
    X_test_num = ((test_df[FEATURE_COL] - num_mean) / num_std).values

    cat_vocabs = {col: build_categorical_vocab(train_df, col) for col in CAT_COLS}
    X_train_cat = np.column_stack([encode_categorical(train_df, col, cat_vocabs[col]) for col in CAT_COLS])
    X_test_cat = np.column_stack([encode_categorical(test_df, col, cat_vocabs[col]) for col in CAT_COLS])

    y_train = train_df["podium"].astype(float).values
    y_test = test_df["podium"].astype(float).values

    return {
        "train_dataset": TabularDataset(X_train_num, X_train_cat, y_train),
        "test_dataset": TabularDataset(X_test_num, X_test_cat, y_test),
        "cat_vocabs": cat_vocabs,
        "num_mean": num_mean,
        "num_std": num_std,
        "y_test": y_test,
    }


def find_best_threshold_np(y_true, y_proba, thresholds=None):
    """
    Stessa identica logica di src.train.find_best_threshold, ma operante su array numpy grezzi invece che su un modello scikit-learn

    Parametri:
        y_true: array delle etichette vere (0/1)
        y_proba: array delle probabilità predette dal modello
        thresholds: soglie da provare; default 0.1-0.9 a step di 0.05

    Ritorna:
        dict con threshold, precision, recall, f1 al punto migliore
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)

    best_f1, best = -1, None
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best = {
                "threshold": t,
                "precision": precision_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred),
                "f1": f1,
            }
    return best

def build_driver_sequences(df, seq_features=None, seq_len=10):
    """
    Costruisce, per ciascuna riga (gara di un pilota), la sequenza delle sue ultime seq_len gare precedenti, con padding a sinistra per chi ha meno storico del massimo

    Parametri:
        df: DataFrame ordinato cronologicamente
        seq_features: colonne grezze da includere nella sequenza; default grid/positionOrder/points
        seq_len: lunghezza massima della sequenza

    Ritorna:
        tupla (sequences, lengths):
        - sequences: array (n_righe, seq_len, n_features), con padding di zeri nelle posizioni iniziali per chi ha meno storico
        - lengths: array (n_righe,) con il numero reale di gare nello storico di ciascuna riga
    """
    if seq_features is None:
        seq_features = ["grid", "positionOrder", "points"]

    n_features = len(seq_features)
    sequences = np.zeros((len(df), seq_len, n_features), dtype=np.float32)
    lengths = np.ones(len(df), dtype=np.int64)

    # df è già ordinato cronologicamente da build_working_dataset groupby preserva quest'ordine all'interno di ciascun gruppo pilota, quindi la storia si costruisce 
    # naturalmente in ordine temporale corretto
    for driver_id, group in df.groupby("driverId"):
        history = []  # storico progressivo, aggiornato gara per gara
        for idx in group.index:
            n_hist = min(len(history), seq_len)
            if n_hist > 0:
                recent = history[-seq_len:]
                # Padding a sinistra: le gare reali occupano le ultime posizioni dell'array, gli zeri iniziali rappresentano "nessuna gara qui"
                # pack_padded_sequence saprà ignorarli grazie a 'lengths'
                sequences[idx, seq_len - n_hist:seq_len, :] = np.array(recent)
                lengths[idx] = n_hist

            # Aggiungiamo la gara corrente allo storico solo ora, dopo aver già costruito la sequenza per questa riga; così le righe future vedranno questa gara, ma questa riga 
            # stessa non la vede mai
            history.append(group.loc[idx, seq_features].values.astype(np.float32))

    return sequences, lengths

def build_race_graphs(df, feature_cols, feat_mean, feat_std):
    """
    Costruisce un grafo per ciascuna gara: nodi = piloti in griglia, archi = relazione compagno di squadra (bidirezionale)

    Parametri:
        df: DataFrame con FEATURE_COL, driverId, constructorId, raceId, podium
        feature_cols: lista delle colonne feature da usare come attributi dei nodi
        feat_mean, feat_std: statistiche di normalizzazione

    Ritorna:
        Lista di oggetti torch_geometric.data.Data, uno per gara
    """
    from torch_geometric.data import Data

    graphs = []

    for race_id, race_group in df.groupby("raceId"):
        race_group = race_group.reset_index(drop=True)

        x = torch.tensor(
            ((race_group[feature_cols] - feat_mean) / feat_std).values,
            dtype=torch.float32
        )
        y = torch.tensor(race_group["podium"].astype(float).values, dtype=torch.float32)

        # Costruzione archi: per ciascuna coppia di piloti con la stessa constructorId in una data gara, creiamo un arco in entrambe le direzioni
        edges = []
        for constructor_id, team_group in race_group.groupby("constructorId"):
            indices = team_group.index.tolist()
            for i in range(len(indices)):
                for j in range(len(indices)):
                    if i != j:
                        edges.append([indices[i], indices[j]])

        if len(edges) == 0:
            # Gara senza nessuna coppia di compagni rilevabile un piccolo grafo senza archi è comunque valido per la GNN, semplicemente
            # non propaga informazione tra nodi in quel caso specifico
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return graphs