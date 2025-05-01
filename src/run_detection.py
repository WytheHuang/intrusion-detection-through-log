import argparse
import sys

import pandas as pd
import torch

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Analyze log files for potential security threats using Gemma 12B")
    parser.add_argument("--input", type=str, required=True, help="Path to the input log file (CSV format)")
    parser.add_argument("--output", type=str, help="Path to save analysis results (CSV format)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument(
        "--threshold", type=float, default=0.7, help="Confidence threshold for threat detection (default: 0.7)"
    )
    return parser


def analyze_logs(
    input_file: str,
    output_file: str | None = None,
    batch_size: int = 8,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Analyze log entries for potential security threats.

    Args:
        input_file: Path to input CSV file containing logs
        output_file: Optional path to save results
        batch_size: Batch size for processing
        threshold: Confidence threshold for threat detection

    Returns:
        DataFrame containing analysis results
    """
    # Initialize model and processor
    model = LogAnalysisModel(model_name="gemma:12b")
    classifier = LogClassifier(model)
    processor = LogProcessor(batch_size=batch_size)

    # Load and process data
    print(f"Loading log data from {input_file}...")
    data = pd.read_csv(input_file)

    # Prepare data loader
    text_column = "log_text" if "log_text" in data.columns else data.columns[0]
    data_loader = processor.prepare_data(input_file, text_column=text_column)[2]  # Use test loader

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
        if row["prediction"] == 1 and row["confidence"] >= threshold
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
    if output_file:
        data.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")

    return data


def main() -> None:
    """Main entry point for the script."""
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        analyze_logs(
            input_file=args.input, output_file=args.output, batch_size=args.batch_size, threshold=args.threshold
        )
    except Exception as e:
        print(f"Error during analysis: {e!s}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
