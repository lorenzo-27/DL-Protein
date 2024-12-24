import argparse
import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AA = [
    "A",
    "C",
    "E",
    "D",
    "G",
    "F",
    "I",
    "H",
    "K",
    "M",
    "L",
    "N",
    "Q",
    "P",
    "S",
    "R",
    "T",
    "W",
    "V",
    "Y",
    "X",
    "NoSeq",
]
SS = ["L", "B", "E", "G", "I", "H", "S", "T", "NoSeq"]

SEQ_LEN = 700
NUM_FEATURES = 57
AMINO_ACID_RESIDUES = 21
NUM_CLASSES = 8
SEED = 42


def split_dataset(data, seed):
    """
    Suddivide il dataset in training (5600), validation (256) e test (272) come da Zhou et al. 2014.
    """
    np.random.seed(seed)
    np.random.shuffle(data)
    train = data[:5600]
    val = data[5605:5861]
    test = data[5861:]
    return train, val, test


def extract_features_and_labels(data):
    """
    Estrae le features e le labels dal dataset. Effettua la trasposizione e il cast dei dati.
    """
    X = data[:, :, :AMINO_ACID_RESIDUES]  # amino acid residues
    X = X.transpose(0, 2, 1)  # da (N, 700, 21) a (N, 21, 700)
    X = X.astype(np.float32)
    y = data[
        :, :, AMINO_ACID_RESIDUES : AMINO_ACID_RESIDUES + NUM_CLASSES
    ]  # 8-state secondary structure
    y = y.transpose(0, 2, 1)  # da (N, 700, 8) a (N, 8, 700)
    y = y.astype(int)
    print(f"Shape di X: {X.shape}. Shape di y: {y.shape}")
    return X, y


def load_data(path, seed):
    """
    Carica il dataset e restituisce i dati e le etichette suddivisi in training, validation e test.
    """
    with gzip.open("data/%s" % (path), "rb") as f:
        ds = np.load(f)
    ds = np.reshape(ds, (-1, SEQ_LEN, NUM_FEATURES))
    data = np.zeros((ds.shape[0], ds.shape[1], AMINO_ACID_RESIDUES + NUM_CLASSES))
    data[:, :, :AMINO_ACID_RESIDUES] = ds[:, :, 35:56]  # amino acid residues
    data[:, :, AMINO_ACID_RESIDUES:] = ds[:, :, 22:30]  # secondary structure labels

    if "cullpdb" in path:
        train, val, test = split_dataset(data, seed)
        train_X, train_y = extract_features_and_labels(train)
        val_X, val_y = extract_features_and_labels(val)
        test_X, test_y = extract_features_and_labels(test)
        print(
            f"Dataset cullpdb caricato. Numero di campioni di training: {train_X.shape[0]}. Numero di campioni di validazione: {val_X.shape[0]}. Numero di campioni di test: {test_X.shape[0]}"
        )
        print(f"Shape di train_X: {train_X.shape}. Shape di train_y: {train_y.shape}")
        print(f"Shape di val_X: {val_X.shape}. Shape di val_y: {val_y.shape}")
        print(f"Shape di test_X: {test_X.shape}. Shape di test_y: {test_y.shape}")
        return train_X, train_y, val_X, val_y, test_X, test_y
    else:
        cb513_X, cb513_y = extract_features_and_labels(data)
        print(f"Dataset CB513 caricato. Numero di campioni: {cb513_X.shape[0]}")
        print(f"Shape di cb513_X: {cb513_X.shape}. Shape di cb513_y: {cb513_y.shape}")
        return cb513_X, cb513_y


def main(args):
    # Ensure figures directory exists
    os.makedirs("figure", exist_ok=True)

    # training set
    train_X, train_y, val_X, val_y, test_X, test_y = load_data(
        args.train_path, seed=SEED
    )
    train_seq_len = (train_X != 0).any(axis=1).sum(axis=1)
    val_seq_len = (val_X != 0).any(axis=1).sum(axis=1)
    test_seq_len = (test_X != 0).any(axis=1).sum(axis=1)
    total_seq_len = np.concatenate([train_seq_len, val_seq_len, test_seq_len])

    print("train %d sequences" % (len(total_seq_len)))

    # test set
    test_X, test_y = load_data(args.test_path, seed=SEED)
    test_seq_len = (test_X != 0).any(axis=1).sum(axis=1)

    print("test %d sequences" % (len(test_seq_len)))

    # Generazione dei grafici
    # Lunghezza delle sequenze
    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    pd.DataFrame(train_seq_len, columns=["training set"]).hist(bins=100, ax=ax)

    ax = plt.subplot(1, 2, 2)
    pd.DataFrame(test_seq_len, columns=["test set"]).hist(bins=100, ax=ax)

    plt.suptitle("Sequence length")
    plt.tight_layout()
    plt.savefig("figure/seq_len.png")

    # amino acid
    out = open("data/train.aa", "w")
    train_aa = []
    for seq, seq_len in zip(train_X, train_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].transpose(1, 0):
            aa = np.where(aa == 1)
            assert len(aa) == 1
            aa = AA[int(aa[0][0])]
            seq_aa.append(aa)
        out.write("".join(seq_aa) + "\n")
        train_aa += seq_aa
    out.close()

    out = open("data/test.aa", "w")
    test_aa = []
    for seq, seq_len in zip(test_X, test_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].transpose(1, 0):
            aa = np.where(aa == 1)
            assert len(aa) == 1
            aa = AA[int(aa[0][0])]
            seq_aa.append(aa)
        out.write("".join(seq_aa) + "\n")
        test_aa += seq_aa
    out.close()

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    df = pd.DataFrame(train_aa).groupby(0).size()
    df.index.name = "training set"
    df.plot(kind="bar", ax=ax)
    ax.grid(True)

    ax = plt.subplot(1, 2, 2)
    df = pd.DataFrame(test_aa).groupby(0).size()
    df.index.name = "test set"
    df.plot(kind="bar", ax=ax)
    ax.grid(True)

    plt.suptitle("Amino Acid Residues")
    plt.tight_layout()
    plt.savefig("figure/amino_acid.png")

    # secondary structure
    out = open("data/train.ss", "w")
    train_ss = []
    for seq, seq_len in zip(train_y, train_seq_len):
        ss = list(map(lambda x: SS[int(x)], seq[:seq_len]))
        out.write("".join(ss) + "\n")
        train_ss += ss
    out.close()

    out = open("data/test.ss", "w")
    test_ss = []
    for seq, seq_len in zip(test_y, test_seq_len):
        ss = list(map(lambda x: SS[int(x)], seq[:seq_len]))
        out.write("".join(ss) + "\n")
        test_ss += ss
    out.close()

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    df = pd.DataFrame(train_ss).groupby(0).size()
    df.index.name = "training set"
    df.plot(kind="bar", ax=ax)
    ax.grid(True)

    ax = plt.subplot(1, 2, 2)
    df = pd.DataFrame(test_ss).groupby(0).size()
    df.index.name = "test set"
    df.plot(kind="bar", ax=ax)
    ax.grid(True)

    plt.suptitle("8-state Secondary Structures")
    plt.tight_layout()
    plt.savefig("figure/ss.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Secondary Structure Prediction"
    )
    parser.add_argument("--train_path", default="cullpdb+profile_6133.npy.gz")
    parser.add_argument("--test_path", default="cb513+profile_split1.npy.gz")
    args = parser.parse_args()
    print(vars(args))

    main(args)
