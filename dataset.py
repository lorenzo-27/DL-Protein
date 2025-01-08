import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import gzip


class ProteinDataset(Dataset):
    def __init__(self, data_path):
        """
        Initialize the dataset by loading the compressed data and processing it.
        """
        with gzip.open(f"data/{data_path}", 'rb') as f:
            self.data = np.load(f)
        self.process_data()

    def process_data(self):
        """Process the raw data while maintaining sequence structure."""
        # Original data shape is (num_proteins, 700, 57)
        self.num_proteins = self.data.shape[0]
        self.seq_length = self.data.shape[1]

        # Extract features
        aa_features = self.data[:, :, 0:20]  # one-hot encoding
        pssm_features = self.data[:, :, 35:55]  # PSSM values

        # Normalize PSSM features per-protein
        pssm_mean = np.mean(pssm_features, axis=(1, 2), keepdims=True)
        pssm_std = np.std(pssm_features, axis=(1, 2), keepdims=True)
        pssm_features = (pssm_features - pssm_mean) / (pssm_std + 1e-8)

        # Add positional encoding
        positions = np.arange(self.seq_length)
        pos_encoding = positions / self.seq_length
        pos_encoding = np.tile(pos_encoding[np.newaxis, :, np.newaxis], (self.num_proteins, 1, 1))

        # Combine all features
        self.features = np.concatenate([aa_features, pssm_features, pos_encoding], axis=2)

        # Extract labels
        self.labels = self.data[:, :, 22:30]

        # Create masks
        self.mask = np.sum(aa_features, axis=2) != 0

    def __len__(self):
        """Return the number of samples (proteins) in the dataset."""
        return self.num_proteins

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        # Get features and transpose to (channels, sequence_length)
        features = torch.tensor(self.features[idx], dtype=torch.float32)  # Shape: (700, 40)
        features = features.transpose(0, 1)  # Shape: (40, 700)

        # Get labels and transpose to match the expected shape
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)  # Shape: (700, 8)
        labels = labels.transpose(0, 1)  # Shape: (8, 700)

        # Get mask
        mask = torch.tensor(self.mask[idx], dtype=torch.bool)  # Shape: (700,)

        return features, labels, mask


def load_data(batch_size):
    """
    Load and prepare the protein sequence datasets.

    Args:
        batch_size: Size of batches for the DataLoader

    Returns:
        train_loader: DataLoader for the training set (CullPDB)
        test_loader: DataLoader for the test set (CB513)
    """
    # Load datasets
    train_dataset = ProteinDataset('cullpdb+profile_6133_filtered.npy.gz')
    test_dataset = ProteinDataset('cb513+profile_split1.npy.gz')

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader

def main():
    # Load data
    batch_size = 32
    train_loader, test_loader = load_data(batch_size)

    # Print dataset information
    print(f"Number of batches in train_loader: {len(train_loader)}")
    print(f"Number of batches in test_loader: {len(test_loader)}")

    # Test a batch
    for features, labels, mask in train_loader:
        print(f"Feature shape: {features.shape}")  # Should be [batch_size, 40, 700]
        print(f"Labels shape: {labels.shape}")  # Should be [batch_size, 8, 700]
        print(f"Mask shape: {mask.shape}")  # Should be [batch_size, 700]
        print(f"Feature type: {features.dtype}")
        print(f"Labels type: {labels.dtype}")
        break


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
