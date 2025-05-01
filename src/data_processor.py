import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LogDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        return {
            "texts": self.texts[idx],
            "labels": torch.tensor(self.labels[idx]),
        }

    def __len__(self) -> int:
        return len(self.labels)


class LogProcessor:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    def prepare_data(
        self, log_file: str, label_column: str = "label", text_column: str = "log_text"
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare log data for training and evaluation"""
        # Read and preprocess the data
        df = pd.read_csv(log_file)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()

        # Split the data
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.3, random_state=42
        )
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        # Create datasets
        train_dataset = LogDataset(train_texts, train_labels)
        val_dataset = LogDataset(val_texts, val_labels)
        test_dataset = LogDataset(test_texts, test_labels)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader
