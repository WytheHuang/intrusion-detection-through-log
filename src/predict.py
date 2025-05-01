"""Log Analysis Prediction Module.

This module provides functionality for analyzing log entries using the trained model
to detect potential intrusions or suspicious activities.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier


class LogPredictor:
    """LogPredictor class for making predictions on log entries.

    A class that loads a trained model and provides methods to make predictions
    on both single and batch log entries for anomaly detection.

    Args:
        model_path (str): Path to the saved model file
        model_name (str, optional): Name of the model architecture to use. Defaults to "gemma3:12b"
        num_labels (int, optional): Number of classification labels. Defaults to 2

    Attributes:
        model (LogAnalysisModel): The loaded model instance
        classifier (LogClassifier): Classifier instance using the loaded model

    Examples:
        >>> predictor = LogPredictor("path/to/model.pth")
        >>> result = predictor.predict_single("Sample log entry")
        >>> results = predictor.predict_batch(["Log 1", "Log 2"])
    """

    def __init__(self, model_path: str, model_name: str = "gemma3:12b", num_labels: int = 2) -> None:
        """Initialize the LogPredictor object.

        This constructor initializes a log analysis predictor by loading a pre-trained model
        and setting up the classifier.

        Args:
            model_path (str): Path to the saved model weights/checkpoint
            model_name (str, optional): Name of the model architecture to use. Defaults to "gemma3:12b"
            num_labels (int, optional): Number of classification labels. Defaults to 2

        Returns:
            None

        Example:
            predictor = LogPredictor(
                model_path="models/checkpoint.pt",
                model_name="gemma3:12b",
                num_labels=2
            )
        """
        self.model = LogAnalysisModel(model_name=model_name, num_labels=num_labels)
        self.model.load(model_path)
        self.classifier = LogClassifier(self.model)

    def predict_single(self, log_entry: str) -> int:
        """Predict a single log entry."""
        batch = {"texts": [log_entry]}
        return self.classifier.predict(batch).item()

    def predict_batch(self, log_entries: list[str]) -> list[int]:
        """Predict multiple log entries."""
        batch = {"texts": log_entries}
        return self.classifier.predict(batch).tolist()


def process_input(input_path: str | None, output_path: str | None, args: argparse.Namespace):
    """Process log data and make predictions using a trained model.

    This function takes log data from either a file or stdin, uses a LogPredictor model to make predictions,
    and outputs the results as JSON either to a file or stdout.

    Args:
        input_path (str | None): Path to input file containing logs. If None, reads from stdin.
        output_path (str | None): Path to output JSON file. If None, prints to stdout.
        args: Argument namespace containing:
            - model_path: Path to the trained model
            - model_name: Name of the model to use
            - num_labels: Number of classification labels

    Returns:
        None: Results are either written to file or printed to stdout

    Example:
        >>> process_input("logs.txt", "results.json", args)
        # Processes logs from logs.txt and saves predictions to results.json

        >>> process_input(None, None, args)
        # Reads from stdin and prints predictions to stdout
    """
    predictor = LogPredictor(args.model_path, model_name=args.model_name, num_labels=args.num_labels)
    # Process input from file or stdin
    if input_path:
        with Path(input_path).open() as f:
            logs = [line.strip() for line in f]
    else:
        logs = [line.strip() for line in sys.stdin]

    # Make predictions
    predictions = predictor.predict_batch(logs)
    results = [
        {
            "log": log,
            "prediction": pred,
            "label": "malicious" if pred == 1 else "benign",
        }
        for log, pred in zip(logs, predictions)
    ]

    # Output results
    output_json = json.dumps(results, indent=2)
    if output_path:
        with Path(output_path).open("w") as f:
            f.write(output_json)
    else:
        print(output_json)


def analyze_logs(model_path: str, log_file: str, output_file: str | None = None, batch_size: int = 16) -> pd.DataFrame:
    """Analyze log entries for potential security threats.

    Args:
        model_path: Path to the trained model checkpoint
        log_file: Path to the log file to analyze
        output_file: Optional path to save results
        batch_size: Batch size for processing

    Returns:
        DataFrame containing log entries and their analysis results
    """
    # Initialize model
    model = LogAnalysisModel(model_name="gemma:12b")
    classifier = LogClassifier(model)
    classifier.model.load(model_path)

    # Prepare data
    processor = LogProcessor(batch_size=batch_size)
    data = pd.read_csv(log_file)

    # Create DataLoader for prediction
    _, _, test_loader = processor.prepare_data(log_file)

    # Make predictions
    all_predictions = []
    for batch in test_loader:
        predictions = classifier.predict(batch)
        all_predictions.extend(predictions.numpy().tolist())

    # Add predictions to DataFrame
    data["prediction"] = all_predictions
    data["threat_level"] = data["prediction"].map({0: "Benign", 1: "Suspicious"})

    # Save results if output path provided
    if output_file:
        data.to_csv(output_file, index=False)

    return data


def main() -> None:
    """Run the log analysis prediction script."""
    parser = argparse.ArgumentParser(description="Analyze log files for potential security threats")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input log file")
    parser.add_argument("--output", type=str, help="Path to save analysis results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing")

    args = parser.parse_args()

    try:
        results = analyze_logs(
            model_path=args.model, log_file=args.input, output_file=args.output, batch_size=args.batch_size
        )

        # Print summary of findings
        total_logs = len(results)
        suspicious_count = len(results[results["prediction"] == 1])
        print("\nAnalysis Complete:")
        print(f"Total log entries analyzed: {total_logs}")
        print(f"Suspicious entries detected: {suspicious_count}")
        print(f"Percentage of suspicious activity: {(suspicious_count / total_logs) * 100:.2f}%")

        if args.output:
            print(f"\nDetailed results saved to: {args.output}")

    except Exception as e:
        print(f"Error during analysis: {e!s}")


if __name__ == "__main__":
    main()
