import os
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

AA = ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X', 'NoSeq']
SS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'NoSeq']

def split_dataset(data):
    """
    Suddivide il dataset in training (5600), validation (256) e test (272) come da Zhou et al. 2014.
    """
    train = data[:5600]
    val = data[5600:5856]
    test = data[5856:]
    return train, val, test

def data_load(path):
    """
    Carica il dataset e restituisce i dati e le etichette suddivisi in training, validation e test.
    """
    with gzip.open('data/%s' % (path), 'rb') as f:
        data = np.load(f)
    data = data.reshape(-1, 700, 57)  # original 57 features

    X = data[:, :, np.arange(21)]  # 20-residues + non-seq
    X = X.transpose(0, 2, 1).astype(np.float32)

    y = data[:, :, 22:30]  # 8-state
    y = np.array([np.dot(yi, np.arange(8)) for yi in y]).astype(int)

    if 'cullpdb' in path:
        X_train, X_val, X_test = split_dataset(X)
        y_train, y_val, y_test = split_dataset(y)
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X, y


def main(args):
    # Ensure figures directory exists
    os.makedirs('figure', exist_ok=True)

    # training set
    train_X, train_y, val_X, val_y, test_X, test_y = data_load(args.train_path)
    train_seq_len = (train_X != 0).any(axis=1).sum(axis=1)
    val_seq_len = (val_X != 0).any(axis=1).sum(axis=1)
    test_seq_len = (test_X != 0).any(axis=1).sum(axis=1)
    total_seq_len = np.concatenate([train_seq_len, val_seq_len, test_seq_len])

    print('train %d sequences' % (len(total_seq_len)))

    # test set
    test_X, test_y = data_load(args.test_path)
    test_seq_len = (test_X != 0).any(axis=1).sum(axis=1)

    print('test %d sequences' % (len(test_seq_len)))

    # Generazione dei grafici
    # Lunghezza delle sequenze
    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    pd.DataFrame(train_seq_len, columns=['training set']).hist(bins=100, ax=ax)

    ax = plt.subplot(1, 2, 2)
    pd.DataFrame(test_seq_len, columns=['test set']).hist(bins=100, ax=ax)

    plt.suptitle('Sequence length')
    plt.tight_layout()
    plt.savefig('figure/seq_len.png')

    # amino acid
    out = open('data/train.aa', 'w')
    train_aa = []
    for seq, seq_len in zip(train_X, train_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].transpose(1, 0):
            aa = np.where(aa == 1)
            assert len(aa) == 1
            aa = AA[int(aa[0][0])]
            seq_aa.append(aa)
        out.write(''.join(seq_aa) + '\n')
        train_aa += seq_aa
    out.close()

    out = open('data/test.aa', 'w')
    test_aa = []
    for seq, seq_len in zip(test_X, test_seq_len):
        seq_aa = []
        for aa in seq[:, :seq_len].transpose(1, 0):
            aa = np.where(aa == 1)
            assert len(aa) == 1
            aa = AA[int(aa[0][0])]
            seq_aa.append(aa)
        out.write(''.join(seq_aa) + '\n')
        test_aa += seq_aa
    out.close()

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    df = pd.DataFrame(train_aa).groupby(0).size()
    df.index.name = 'training set'
    df.plot(kind='bar', ax=ax)
    ax.grid(True)

    ax = plt.subplot(1, 2, 2)
    df = pd.DataFrame(test_aa).groupby(0).size()
    df.index.name = 'test set'
    df.plot(kind='bar', ax=ax)
    ax.grid(True)

    plt.suptitle('Amino Acid Residues')
    plt.tight_layout()
    plt.savefig('figure/amino_acid.png')

    # secondary structure
    out = open('data/train.ss', 'w')
    train_ss = []
    for seq, seq_len in zip(train_y, train_seq_len):
        ss = list(map(lambda x: SS[int(x)], seq[:seq_len]))
        out.write(''.join(ss) + '\n')
        train_ss += ss
    out.close()

    out = open('data/test.ss', 'w')
    test_ss = []
    for seq, seq_len in zip(test_y, test_seq_len):
        ss = list(map(lambda x: SS[int(x)], seq[:seq_len]))
        out.write(''.join(ss) + '\n')
        test_ss += ss
    out.close()

    plt.figure(figsize=(12, 4))

    ax = plt.subplot(1, 2, 1)
    df = pd.DataFrame(train_ss).groupby(0).size()
    df.index.name = 'training set'
    df.plot(kind='bar', ax=ax)
    ax.grid(True)

    ax = plt.subplot(1, 2, 2)
    df = pd.DataFrame(test_ss).groupby(0).size()
    df.index.name = 'test set'
    df.plot(kind='bar', ax=ax)
    ax.grid(True)

    plt.suptitle('8-state Secondary Structures')
    plt.tight_layout()
    plt.savefig('figure/ss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protein Secondary Structure Prediction')
    parser.add_argument('--train_path', default='cullpdb+profile_6133.npy.gz')
    parser.add_argument('--test_path', default='cb513+profile_split1.npy.gz')
    args = parser.parse_args()
    print(vars(args))

    main(args)