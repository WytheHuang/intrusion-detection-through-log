import argparse
import dataclasses
import sys
from typing import Literal

import pandas as pd
import torch

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier

ProviderType = Literal["ollama", "openai", "gemini"]


@dataclasses.dataclass
class AnalysisConfig:
    """Configuration for log analysis."""

    input_file: str
    output_file: str | None = None
    batch_size: int = 8
    threshold: float = 0.7
    provider: ProviderType = "ollama"
    model_name: str | None = None
    api_key: str | None = None


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze log files for potential security threats")
    parser.add_argument("--input", type=str, required=True, help="Path to the input log file (CSV format)")
    parser.add_argument("--output", type=str, help="Path to save analysis results (CSV format)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Confidence threshold for threat detection (default: 0.7)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "gemini"],
        default="ollama",
        help="LLM provider to use (default: ollama)",
    )
    parser.add_argument("--model-name", type=str, help="Model name for the chosen provider")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI or Gemini (not needed for Ollama)")
    return parser


def analyze_logs(config: AnalysisConfig) -> pd.DataFrame:
    """Analyze log entries for potential security threats.

    Args:
        config: Analysis configuration parameters

    Returns:
        DataFrame containing analysis results
    """
    # Initialize model and processor
    model = LogAnalysisModel(provider=config.provider, model_name=config.model_name, api_key=config.api_key)
    classifier = LogClassifier(model)
    processor = LogProcessor(batch_size=config.batch_size)

    # Load and process data
    print(f"Loading log data from {config.input_file}...")
    data = pd.read_csv(config.input_file)

    # Prepare data loader
    text_column = "log_text" if "log_text" in data.columns else data.columns[0]
    data_loader = processor.prepare_data(config.input_file, text_column=text_column)[2]  # Use test loader

    # Make predictions
    print("Analyzing logs for potential threats...")
    all_predictions = []
    all_confidence_scores = []

    with torch.no_grad():
        for batch in data_loader:
            # Get model predictions and confidence scores
            logits = classifier.get_logits(batch)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            confidence = torch.max(probs, dim=1)[0]

            all_predictions.extend(predictions.cpu().numpy())
            all_confidence_scores.extend(confidence.cpu().numpy())

    # Add predictions to DataFrame
    data["prediction"] = all_predictions
    data["confidence"] = all_confidence_scores
    data["threat_level"] = data.apply(
        lambda row: "High Risk"
        if row["prediction"] == 1 and row["confidence"] >= config.threshold
        else ("Suspicious" if row["prediction"] == 1 else "Benign"),
        axis=1,
    )

    # Generate summary
    total_logs = len(data)
    high_risk = len(data[data["threat_level"] == "High Risk"])
    suspicious = len(data[data["threat_level"] == "Suspicious"])

    print("\nAnalysis Summary:")
    print(f"Total log entries analyzed: {total_logs}")
    print(f"High risk entries: {high_risk} ({(high_risk / total_logs) * 100:.2f}%)")
    print(f"Suspicious entries: {suspicious} ({(suspicious / total_logs) * 100:.2f}%)")

    # Save results if output path provided
    if config.output_file:
        data.to_csv(config.output_file, index=False)
        print(f"\nDetailed results saved to: {config.output_file}")

    return data


def main() -> None:
    """Main entry point for the script."""
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        config = AnalysisConfig(
            input_file=args.input,
            output_file=args.output,
            batch_size=args.batch_size,
            threshold=args.threshold,
            provider=args.provider,
            model_name=args.model_name,
            api_key=args.api_key,
        )
        analyze_logs(config)
    except Exception as e:
        print(f"Error during analysis: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
