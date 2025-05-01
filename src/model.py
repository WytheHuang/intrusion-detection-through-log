import re
from datetime import datetime
from typing import Any

import numpy as np
import ollama
import torch
from torch import nn


class LogAnalysisModel(nn.Module):
    """A neural network model for analyzing log entries using Ollama embeddings.

    This model uses Ollama's Gemma 12B model to analyze system logs for intrusion detection.
    It preprocesses log entries to extract key information and generates embeddings for classification.

    Attributes:
        num_labels : int
            Number of classification labels (default is 2 for binary classification: benign vs malicious)
        model_name : str
            Name of the Ollama model to use for embeddings
        client : ollama.Client
            Client instance for making requests to Ollama API
    """

    def __init__(self, model_name: str = "gemma:12b", num_labels: int = 2) -> None:
        """Initialize the model class.

        This constructor initializes a new instance with a specified LLM model and number of output labels.

        Args:
            model_name (str): Name of the Ollama model to use. Defaults to "gemma:12b".
            num_labels (int): Number of classification labels. Defaults to 2.

        Attributes:
            num_labels (int): Number of classification labels stored as instance variable
            model_name (str): Name of the model stored as instance variable
            client (ollama.Client): Ollama client instance for model interaction
        """
        super().__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.client = ollama.Client()

    def _preprocess_log(self, log_entry: str) -> str:
        """Preprocess a log entry to extract relevant information.

        This method:
        1. Extracts timestamps if present
        2. Identifies IP addresses and ports
        3. Looks for common security-relevant keywords
        4. Normalizes the text for better analysis

        Args:
            log_entry: The raw log entry text

        Returns:
            Preprocessed log entry optimized for security analysis
        """
        # Extract timestamp if present
        timestamp_pattern = r"\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}"
        timestamp_match = re.search(timestamp_pattern, log_entry)
        timestamp = ""
        if timestamp_match:
            try:
                dt = datetime.strptime(timestamp_match.group(), "%Y-%m-%d %H:%M:%S")  # noqa: DTZ007
                timestamp = f"[TIME:{dt.strftime('%H:%M:%S')}]"
            except ValueError:
                pass

        # Extract IP addresses
        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
        ips = re.findall(ip_pattern, log_entry)
        ip_info = f"[IPs:{','.join(ips)}]" if ips else ""

        # Identify security-relevant keywords
        security_keywords = [
            "failed",
            "error",
            "denied",
            "unauthorized",
            "invalid",
            "attack",
            "breach",
            "suspicious",
            "warning",
            "critical",
        ]
        found_keywords = [kw for kw in security_keywords if kw in log_entry.lower()]
        keyword_info = f"[KEYWORDS:{','.join(found_keywords)}]" if found_keywords else ""

        # Normalize text
        normalized_text = re.sub(r"\s+", " ", log_entry).strip()

        # Construct analysis prompt
        return f"Analyze this log entry for potential security threats: {timestamp} {ip_info} {keyword_info} {normalized_text}"

    def _process_text(self, text: str) -> np.ndarray:
        """Process text through Ollama and return embeddings.

        Args:
            text: The preprocessed log entry text

        Returns:
            Embedding vector for the input text
        """
        # Preprocess the log entry
        processed_text = self._preprocess_log(text)

        # Get embeddings using Ollama
        response = self.client.embeddings(model=self.model_name, prompt=processed_text)
        return np.array(response["embedding"])

    def forward(self, texts: list[str]) -> torch.Tensor:
        """Process a batch of texts through the model.

        Parameters:
            texts : list[str]
                List of text entries to process

        Returns:
            torch.Tensor
                Tensor of embeddings for the input texts
        """
        # Process each text through Ollama and get embeddings
        embeddings = [self._process_text(text) for text in texts]
        return torch.tensor(np.stack(embeddings), dtype=torch.float32)

    def save(self, path: str) -> None:
        """Save model parameters and configuration."""
        torch.save(
            {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "state_dict": self.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model parameters and configuration."""
        checkpoint = torch.load(path)
        self.model_name = checkpoint["model_name"]
        self.num_labels = checkpoint["num_labels"]
        self.load_state_dict(checkpoint["state_dict"])


class LogClassifier:
    """A classifier that uses LogAnalysisModel embeddings for log entry classification.

    This classifier takes embeddings from a LogAnalysisModel and applies a linear
    classification layer to determine if a log entry represents an intrusion.

    Attributes:
        model : LogAnalysisModel
            The model used to get embeddings
        device : str
            The device to run computations on ('cuda' or 'cpu')
        classifier : nn.Linear
            The linear classification layer
        criterion : nn.CrossEntropyLoss
            The loss function
        optimizer : torch.optim.AdamW
            The optimizer for training
    """

    def __init__(
        self,
        model: LogAnalysisModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the LogClassifier.

        Parameters:
            model : LogAnalysisModel
                The model to use for getting embeddings
            device : str
                The device to run computations on (default is 'cuda' if available, else 'cpu')
        """
        self.model = model
        self.device = device
        self.classifier = nn.Linear(4096, model.num_labels).to(device)  # Ollama embeddings are 4096-dimensional
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=2e-5)

    def train_step(self, batch: dict[str, Any]) -> float:
        """Perform a single training step.

        Parameters:
            batch : dict[str, Any]
                A dictionary containing 'texts' and 'labels' for training

        Returns:
            float
                The loss value for this training step
        """
        self.classifier.train()
        self.optimizer.zero_grad()

        texts = batch["texts"]
        labels = batch["labels"].to(self.device)

        embeddings = self.model(texts).to(self.device)
        outputs = self.classifier(embeddings)
        loss = self.criterion(outputs, labels)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the model on a batch of data.

        Parameters:
            batch : dict[str, Any]
                A dictionary containing 'texts' and 'labels' for evaluation

        Returns:
            dict[str, Any]
                A dictionary containing 'loss', 'predictions', and 'labels'
        """
        self.classifier.eval()
        with torch.no_grad():
            texts = batch["texts"]
            labels = batch["labels"].to(self.device)

            embeddings = self.model(texts).to(self.device)
            outputs = self.classifier(embeddings)
            loss = self.criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            return {
                "loss": loss.item(),
                "predictions": predictions.cpu(),
                "labels": labels.cpu(),
            }

    def predict(self, batch: dict[str, Any]) -> torch.Tensor:
        """Make predictions on a batch of data.

        Parameters:
            batch : dict[str, Any]
                A dictionary containing 'texts' for prediction

        Returns:
            torch.Tensor
                The predicted class indices
        """
        self.classifier.eval()
        with torch.no_grad():
            texts = batch["texts"]
            embeddings = self.model(texts).to(self.device)
            outputs = self.classifier(embeddings)
            return torch.argmax(outputs, dim=1).cpu()

    def get_logits(self, batch: dict[str, Any]) -> torch.Tensor:
        """Get raw logits for a batch of data.

        Parameters:
            batch : dict[str, Any]
                A dictionary containing 'texts' for prediction

        Returns:
            torch.Tensor
                The raw logits from the classifier
        """
        self.classifier.eval()
        with torch.no_grad():
            texts = batch["texts"]
            embeddings = self.model(texts).to(self.device)
            return self.classifier(embeddings)
