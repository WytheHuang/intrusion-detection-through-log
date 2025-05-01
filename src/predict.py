import argparse
import json
import os
import sys

from model import LogAnalysisModel, LogClassifier


class LogPredictor:
    def __init__(self, model_path: str, model_name: str = "gemma3:12b", num_labels: int = 2):
        self.model = LogAnalysisModel(model_name=model_name, num_labels=num_labels)
        self.model.load(model_path)
        self.classifier = LogClassifier(self.model)

    def predict_single(self, log_entry: str) -> int:
        """Predict a single log entry"""
        batch = {"texts": [log_entry]}
        return self.classifier.predict(batch).item()

    def predict_batch(self, log_entries: list[str]) -> list[int]:
        """Predict multiple log entries"""
        batch = {"texts": log_entries}
        return self.classifier.predict(batch).tolist()


def process_input(input_path: str | None, output_path: str | None, args):
    predictor = LogPredictor(args.model_path, model_name=args.model_name, num_labels=args.num_labels)

    # Process input from file or stdin
    if input_path:
        with open(input_path) as f:
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
        with open(output_path, "w") as f:
            f.write(output_json)
    else:
        print(output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    parser.add_argument(
        "--input_path",
        type=str,
        help="Path to input file with log entries (one per line). If not provided, reads from stdin",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output JSON file. If not provided, prints to stdout",
    )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Number of classes (default: 2 for binary classification)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemma3:12b",
        help="Name of the Ollama model to use",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}", file=sys.stderr)
        sys.exit(1)

    process_input(args.input_path, args.output_path, args)
