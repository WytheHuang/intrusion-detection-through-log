"""Data generation module for creating synthetic log data.

This module supports multiple LLM providers (Ollama, OpenAI, Gemini) to generate synthetic log data
for training the intrusion detection system. It creates both benign and malicious log entries.
"""

import random
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import ollama
import pandas as pd
from google.generativeai import GenerativeModel
from google.generativeai import configure as configure_genai
from openai import OpenAI
from tqdm import tqdm

ProviderType = Literal["ollama", "openai", "gemini"]


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text using the LLM provider.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Generated text response
        """


class OllamaProvider(BaseProvider):
    """Ollama-based LLM provider."""

    def __init__(self, model_name: str = "gemma3:12b") -> None:
        """Initialize Ollama provider.

        Args:
            model_name: Name of the Ollama model to use
        """
        self.client = ollama.Client()
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated text response
        """
        response = self.client.generate(model=self.model_name, prompt=prompt)
        return response.response.strip()


class OpenAIProvider(BaseProvider):
    """OpenAI-based LLM provider."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo") -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate text using OpenAI.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated text response
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=100,
        )
        return response.choices[0].message.content.strip()


class GeminiProvider(BaseProvider):
    """Google's Gemini-based LLM provider."""

    def __init__(self, api_key: str, model_name: str = "gemini-pro") -> None:
        """Initialize Gemini provider.

        Args:
            api_key: Google API key
            model_name: Name of the Gemini model to use
        """
        configure_genai(api_key=api_key)
        self.model = GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        """Generate text using Gemini.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated text response
        """
        response = self.model.generate_content(prompt)
        return response.text.strip()


class LogGenerator:
    """Generator class for creating synthetic log data using LLMs."""

    def __init__(
        self,
        provider: ProviderType = "ollama",
        model_name: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the log generator.

        Args:
            provider: The LLM provider to use ("ollama", "openai", or "gemini")
            model_name: Name of the model to use (provider-specific)
            api_key: API key for OpenAI or Gemini (not needed for Ollama)
        """
        if provider == "ollama":
            self.provider = OllamaProvider(model_name or "gemma3:12b")
        elif provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required")
            self.provider = OpenAIProvider(api_key, model_name or "gpt-3.5-turbo")
        elif provider == "gemini":
            if not api_key:
                raise ValueError("Google API key is required")
            self.provider = GeminiProvider(api_key, model_name or "gemini-pro")
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _generate_prompt(self, is_malicious: bool) -> str:
        """Create a prompt for log generation.

        Args:
            is_malicious: Whether to generate a malicious or benign log entry

        Returns:
            A prompt string for the LLM
        """
        base_prompt = "Generate a single line system log entry that is "
        if is_malicious:
            return base_prompt + (
                "indicative of a security threat or attack. Include typical indicators "
                "like failed login attempts, unusual file access, or suspicious commands. "
                "Use realistic timestamps, IP addresses, and user information."
            )
        return base_prompt + (
            "representative of normal system activity. Include typical elements "
            "like successful logins, routine file access, or standard system operations. "
            "Use realistic timestamps, IP addresses, and user information."
        )

    def generate_log(self, is_malicious: bool = False) -> dict[str, Any]:
        """Generate a single log entry.

        Args:
            is_malicious: Whether to generate a malicious or benign log entry

        Returns:
            A dictionary containing the generated log and its label
        """
        prompt = self._generate_prompt(is_malicious)
        log_text = self.provider.generate(prompt)

        return {
            "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
            "log_text": log_text,
            "label": 1 if is_malicious else 0,
        }

    def generate_dataset(
        self,
        num_samples: int = 1000,
        malicious_ratio: float = 0.3,
        output_file: str | None = None,
    ) -> pd.DataFrame:
        """Generate a dataset of log entries.

        Args:
            num_samples: Number of log entries to generate
            malicious_ratio: Ratio of malicious to benign entries
            output_file: Optional path to save the dataset

        Returns:
            DataFrame containing the generated logs
        """
        num_malicious = int(num_samples * malicious_ratio)
        num_benign = num_samples - num_malicious

        print("Generating benign logs...")
        logs = [self.generate_log(is_malicious=False) for _ in tqdm(range(num_benign))]

        print("Generating malicious logs...")
        logs.extend([self.generate_log(is_malicious=True) for _ in tqdm(range(num_malicious))])

        # Shuffle the logs
        random.shuffle(logs)
        df = pd.DataFrame(logs)

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Dataset saved to {output_file}")

        return df


def main() -> None:
    """Generate a synthetic dataset for training."""
    generator = LogGenerator()
    output_path = Path("data/synthetic_logs.csv")
    output_path.parent.mkdir(exist_ok=True)

    print("Generating synthetic log dataset...")
    generator.generate_dataset(
        num_samples=10,
        malicious_ratio=0.3,
        output_file=str(output_path),
    )
    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
