import argparse
import gzip
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

SEQ_LEN = 700
NUM_FEATURES = 57

def load_dataset(path):
    with gzip.open(f"data/{path}", "rb") as f:
        ds = np.load(f)
    ds = np.reshape(ds, (-1, SEQ_LEN, NUM_FEATURES))

    # Rimozione features solubilità, terminali N e C, "X" e no_seq
    ds = np.delete(ds, [20, 21, 30, 31, 32, 33, 34, 54, 56], axis=2)

    # Conversione in tensori
    ds = torch.from_numpy(ds)

    # Features (X) e labels (y)
    residues_1h = ds[:, :, :20].type(torch.float)       # One-Hot Amino acid residues - (N, L, 20)
    residues_int = torch.argmax(residues_1h, dim=2)     # Int amino acid residues - (N, L)
    pssm = ds[:, :, 28:].type(torch.float)              # PSSMs - (N, L, 20)
    labels = ds[:, :, 20:28]                            # Secondary structure labels - (N, L, 8)

    # Concatenazione di residues_int e pssm - (N, L, 21)
    X = torch.cat((residues_int.unsqueeze(-1), pssm), dim=-1)

    # y: etichette
    y = labels

    # Transpose X and y to (N, C, L)
    X = X.transpose(1, 2)
    y = y.transpose(1, 2)

    return X, y


def load_data(batch_size):
    cullpdb_X, cullpdb_y = load_dataset("cullpdb+profile_6133_filtered.npy.gz")
    cb513_X, cb513_y = load_dataset("cb513+profile_split1.npy.gz")

    train_loader = DataLoader(TensorDataset(cullpdb_X, cullpdb_y), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(cb513_X, cb513_y), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def main(args):
    # Load the datasets
    train_loader, test_loader = load_data(8)
    print(f"Train set: {len(train_loader.dataset)}, Test set: {len(test_loader.dataset)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Protein Secondary Structure Prediction"
    )
    parser.add_argument("--train_path", default="cullpdb+profile_6133_filtered.npy.gz")
    parser.add_argument("--test_path", default="cb513+profile_split1.npy.gz")
    args = parser.parse_args()
    print(vars(args))

    main(args)
