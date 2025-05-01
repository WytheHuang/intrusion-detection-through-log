import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LogDataset(Dataset):
    """A custom Dataset class for handling log data.

    This class implements a PyTorch Dataset for processing text logs and their corresponding labels.
    It enables efficient data loading and batch processing for model training.

    Args:
        texts (list[str]): A list of text strings containing the log messages
        labels (list[int]): A list of integer labels corresponding to each log message

    Returns:
        dict: A dictionary containing the text and label tensor for the requested index

    Example:
        >>> dataset = LogDataset(["log1", "log2"], [0, 1])
        >>> sample = dataset[0]
        >>> print(sample)
        {'texts': 'log1', 'labels': tensor(0)}
    """

    def __init__(self, texts: list[str], labels: list[int]) -> None:
        """Initialize the data processor with texts and their corresponding labels.

        Args:
            texts (list[str]): List of text samples to be processed.
            labels (list[int]): List of integer labels corresponding to each text sample.
        """
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx: int) -> dict:
        """Get item at the specified index.

        This method implements the subscription access for the dataset, allowing indexing
        operations to retrieve specific examples.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing:
                - texts: The text data at the specified index
                - labels: A torch tensor containing the label data at the specified index
        """
        return {
            "texts": self.texts[idx],
            "labels": torch.tensor(self.labels[idx]),
        }

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset, equal to the length of labels.
        """
        return len(self.labels)


class LogProcessor:
    """Log data processing class for preparing training, validation, and test data.

    This class handles the processing of log data for machine learning tasks, including
    data loading, splitting, and preparing DataLoader objects for training.

    Args:
        batch_size (int, optional): The batch size to use for DataLoaders. Defaults to 16.

    Example:
        >>> processor = LogProcessor(batch_size=32)
        >>> train_loader, val_loader, test_loader = processor.prepare_data('logs.csv')

    Methods:
        prepare_data: Prepares the log data by splitting it into train/val/test sets and creating DataLoaders.

    """

    def __init__(self, batch_size: int = 16) -> None:
        """Initialize the DataProcessor instance.

        Parameters:
            batch_size (int, optional): The size of each batch for data processing. Default is 16.

        Returns:
            None
        """
        self.batch_size = batch_size

    def prepare_data(
        self, log_file: str, label_column: str = "label", text_column: str = "log_text"
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare log data for training and evaluation."""
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
