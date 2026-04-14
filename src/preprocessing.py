"""
Data loading and preprocessing for QuickDraw dataset.

Handles loading of .npy files containing raw drawing data,
normalization, reshaping for neural network input, and
creation of train/test DataLoaders with stratified splitting.
"""
import numpy as np
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class Preprocessor:
    """
    Loads .npy files and prepares DataLoaders for training.

    Args:
        classes: List of class names to load (e.g., ['apple', 'star'])
        max_samples: Maximum samples per class to load
        data_path: Directory containing .npy files
    """
    def __init__(self, classes, max_samples, data_path="data"):
        self.classes = classes
        self.data_path = data_path
        self.max_samples = max_samples

    def load_and_preprocess(self):
        """
        Load numpy files, normalize pixel values, and reshape for training.

        Loads each class .npy file, limits to max_samples, normalizes
        pixel values to [0, 1], and reshapes to (N, 1, 28, 28) for CNN input.

        Returns:
            X: Array of shape (total_samples, 1, 28, 28)
            y: Array of class labels
        """
        all_data = []
        all_labels = []

        print(f"Loading {len(self.classes)} classes...")

        for idx, name in enumerate(self.classes):
            file_path = os.path.join(self.data_path, f"{name}.npy")

            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue

            data = np.load(file_path)[:self.max_samples]

            data = data.astype('float32') / 255.0
            labels = np.full(data.shape[0], idx, dtype=np.int64)
          
            all_data.append(data)
            all_labels.append(labels)
            print(f"  {name}: {data.shape[0]} Samples, Label={idx}")

        if not all_data:
            raise ValueError("No data loaded!\n Please Check paths.")
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        X = X.reshape(-1, 1, 28, 28)
        print(f"Done: X={X.shape}, y={y.shape}")
        return X, y

    def get_loaders(self, test_size=0.2, random_state=42, batch_size=32):
        """
        Split data into train/test sets and return DataLoaders.

        Uses stratified split to ensure all classes are equally represented
        in both train and test sets. Shuffles training data for each epoch.

        Args:
            test_size: Fraction of data to use for testing (default: 0.2)
            random_state: Random seed for reproducibility
            batch_size: Number of samples per batch

        Returns:
            train_loader: DataLoader with shuffled training data
            test_loader: DataLoader with sequential test data
        """
        X, y = self.load_and_preprocess()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
          
        )
       

        train_dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test),
            torch.tensor(y_test)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader