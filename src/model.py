import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar, Literal

import numpy as np
import ollama
import torch
from google.generativeai import GenerativeModel
from google.generativeai import configure as configure_genai
from openai import OpenAI
from torch import nn

ProviderType = Literal["ollama", "openai", "gemini"]


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    embedding_dim: ClassVar[int]

    @abstractmethod
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings for the input text.

        Args:
            text: Input text to get embeddings for

        Returns:
            Embedding vector as numpy array
        """


class OllamaProvider(BaseLLMProvider):
    """Ollama-based LLM provider."""

    embedding_dim = 4096  # Ollama's default embedding dimension

    def __init__(self, model_name: str = "gemma3:12b") -> None:
        """Initialize Ollama provider.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.client = ollama.Client()
        self.model_name = model_name

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using Ollama.

        Args:
            text: Input text to get embeddings for

        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embeddings(model=self.model_name, prompt=text)
        return np.array(response["embedding"])


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-based LLM provider."""

    embedding_dim = 3072  # OpenAI text-embedding-3-large dimension

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-large") -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using OpenAI.

        Args:
            text: Input text to get embeddings for

        Returns:
            Embedding vector as numpy array
        """
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )
        return np.array(response.data[0].embedding)


class GeminiProvider(BaseLLMProvider):
    """Google's Gemini-based LLM provider."""

    embedding_dim = 768  # Gemini's embedding dimension

    def __init__(self, api_key: str, model_name: str = "embedding-001") -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key
            model_name: Name of the Gemini model to use
        """
        configure_genai(api_key=api_key)
        self.model = GenerativeModel(model_name)

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get embeddings using Gemini.

        Args:
            text: Input text to get embeddings for

        Returns:
            Embedding vector as numpy array
        """
        response = self.model.embed_content(text)
        return np.array(response.embedding)


class LogAnalysisModel(nn.Module):
    """A neural network model for analyzing log entries using various LLM providers.

    This model can use different LLM providers (Ollama, OpenAI, Gemini) to analyze
    system logs for intrusion detection. It preprocesses log entries to extract key
    information and generates embeddings for classification.

    Args:
        provider: The LLM provider to use ("ollama", "openai", or "gemini")
        model_name: Name of the model to use (provider-specific)
        api_key: API key for OpenAI or Gemini (not needed for Ollama)
        num_labels: Number of classification labels (default: 2)

    Attributes:
        num_labels : int
            Number of classification labels
        provider : BaseLLMProvider
            The LLM provider instance for generating embeddings
    """

    def __init__(
        self,
        provider: ProviderType = "ollama",
        model_name: str | None = None,
        api_key: str | None = None,
        num_labels: int = 2,
    ) -> None:
        """Initialize the model class.

        Args:
            provider: The LLM provider to use
            model_name: Name of the model to use (provider-specific)
            api_key: API key for OpenAI or Gemini
            num_labels: Number of classification labels
        """
        super().__init__()
        self.num_labels = num_labels

        # Initialize the appropriate provider
        if provider == "ollama":
            self.provider = OllamaProvider(model_name or "gemma3:12b")
        elif provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.provider = OpenAIProvider(api_key, model_name or "text-embedding-3-large")
        elif provider == "gemini":
            if not api_key:
                raise ValueError("Google API key is required")
            self.provider = GeminiProvider(api_key, model_name or "embedding-001")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for the current provider."""
        return self.provider.embedding_dim

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
        """Process text through the LLM provider and return embeddings.

        Args:
            text: The preprocessed log entry text

        Returns:
            Embedding vector for the input text
        """
        # Preprocess the log entry
        processed_text = self._preprocess_log(text)

        # Get embeddings using the provider
        return self.provider.get_embeddings(processed_text)

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
        # Use the provider's embedding dimension
        self.classifier = nn.Linear(model.embedding_dim, model.num_labels).to(device)
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
