"""Intrusion Detection Agent for real-time log analysis.

This module implements an agent that continuously monitors and analyzes log entries
for potential security threats using the trained model.
"""

import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier


class LogAnalysisAgent:
    """Agent for real-time log file monitoring and analysis.

    This class implements an agent that watches for changes in log files and
    analyzes new entries for potential security threats in real-time.

    Args:
        model_path: Path to the trained model checkpoint
        watch_path: Path to the directory or file to monitor
        batch_size: Number of logs to process at once
        threshold: Confidence threshold for threat detection
        output_dir: Directory to save analysis results

    Attributes:
        model: The loaded log analysis model
        classifier: The classifier for making predictions
        processor: Log data processor instance
        watch_path: Path being monitored
        threshold: Detection threshold
        output_dir: Results output directory
    """

    def __init__(
        self,
        model_path: str,
        watch_path: str,
        batch_size: int = 8,
        threshold: float = 0.7,
        output_dir: str | None = None,
    ) -> None:
        """Initialize the log analysis agent.

        Args:
            model_path: Path to the trained model checkpoint
            watch_path: Path to monitor for log files
            batch_size: Batch size for processing
            threshold: Confidence threshold for threat detection
            output_dir: Directory to save results (optional)
        """
        # Initialize model and components
        self.model = LogAnalysisModel(model_name="gemma:12b")
        self.model.load(model_path)
        self.classifier = LogClassifier(self.model)
        self.processor = LogProcessor(batch_size=batch_size)

        # Set paths and parameters
        self.watch_path = Path(watch_path)
        self.threshold = threshold
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging for the agent."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("LogAnalysisAgent")

    async def process_file(self, file_path: Path) -> None:
        """Process a log file and analyze its entries.

        Args:
            file_path: Path to the log file to analyze
        """
        try:
            # Read and process the file
            data = pd.read_csv(file_path)
            text_column = "log_text" if "log_text" in data.columns else data.columns[0]
            data_loader = self.processor.prepare_data(str(file_path), text_column=text_column)[2]

            # Make predictions
            all_predictions = []
            all_confidence_scores = []

            with torch.no_grad():
                for batch in data_loader:
                    # Get predictions and confidence scores
                    logits = self.classifier.get_logits(batch)
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
                if row["prediction"] == 1 and row["confidence"] >= self.threshold
                else ("Suspicious" if row["prediction"] == 1 else "Benign"),
                axis=1,
            )

            # Log high-risk entries
            high_risk_entries = data[data["threat_level"] == "High Risk"]
            if not high_risk_entries.empty:
                self.logger.warning(f"Found {len(high_risk_entries)} high-risk entries!")
                for _, entry in high_risk_entries.iterrows():
                    self.logger.warning(
                        f"High-risk entry detected (confidence: {entry['confidence']:.2f}): {entry[text_column]}"
                    )

            # Save results if output directory is specified
            if self.output_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
                output_file = self.output_dir / f"analysis_{timestamp}.csv"
                data.to_csv(output_file, index=False)
                self.logger.info(f"Analysis results saved to {output_file}")

            # Generate and save summary report
            self._generate_summary_report(data, file_path)

        except Exception as e:
            self.logger.exception(f"Error processing file {file_path}: {e!s}")  # noqa: TRY401

    def _generate_summary_report(self, data: pd.DataFrame, file_path: Path) -> None:
        """Generate a summary report of the analysis.

        Args:
            data: DataFrame containing analysis results
            file_path: Path to the analyzed file
        """
        total_logs = len(data)
        high_risk = len(data[data["threat_level"] == "High Risk"])
        suspicious = len(data[data["threat_level"] == "Suspicious"])

        summary = {
            "timestamp": datetime.now().isoformat(),  # noqa: DTZ005
            "file_analyzed": str(file_path),
            "total_logs": total_logs,
            "high_risk_count": high_risk,
            "high_risk_percentage": (high_risk / total_logs) * 100,
            "suspicious_count": suspicious,
            "suspicious_percentage": (suspicious / total_logs) * 100,
        }

        # Log summary
        self.logger.info("\nAnalysis Summary:")
        self.logger.info(f"Total log entries analyzed: {total_logs}")
        self.logger.info(f"High risk entries: {high_risk} ({summary['high_risk_percentage']:.2f}%)")
        self.logger.info(f"Suspicious entries: {suspicious} ({summary['suspicious_percentage']:.2f}%)")

        # Save summary if output directory is specified
        if self.output_dir:
            summary_file = self.output_dir / "analysis_summary.json"
            if summary_file.exists():
                with summary_file.open("r") as f:
                    summaries = json.load(f)
            else:
                summaries = []

            summaries.append(summary)
            with summary_file.open("w") as f:
                json.dump(summaries, f, indent=2)

    def alert(self, message: str) -> None:
        """Send an alert about potential security threats.

        This method can be extended to integrate with external alerting systems.

        Args:
            message: The alert message to send
        """
        self.logger.warning(f"SECURITY ALERT: {message}")


class LogFileHandler(FileSystemEventHandler):
    """File system event handler for log file changes.

    This handler processes new or modified log files using the analysis agent.

    Args:
        agent: The LogAnalysisAgent instance to use for processing
    """

    def __init__(self, agent: LogAnalysisAgent) -> None:
        """Initialize the file handler."""
        self.agent = agent

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events.

        Args:
            event: The file system event that triggered the handler
        """
        if event.is_directory:
            return

        if event.src_path.endswith(".csv"):
            asyncio.run(self.agent.process_file(Path(event.src_path)))

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation events.

        Args:
            event: The file system event that triggered the handler
        """
        if event.is_directory:
            return

        if event.src_path.endswith(".csv"):
            asyncio.run(self.agent.process_file(Path(event.src_path)))


async def run_agent(
    model_path: str,
    watch_path: str,
    batch_size: int = 8,
    threshold: float = 0.7,
    output_dir: str | None = None,
) -> None:
    """Run the log analysis agent.

    This function sets up and runs the agent to monitor log files for threats.

    Args:
        model_path: Path to the trained model checkpoint
        watch_path: Path to monitor for log files
        batch_size: Batch size for processing
        threshold: Confidence threshold for threat detection
        output_dir: Directory to save results (optional)
    """
    # Initialize agent
    agent = LogAnalysisAgent(
        model_path=model_path,
        watch_path=watch_path,
        batch_size=batch_size,
        threshold=threshold,
        output_dir=output_dir,
    )

    # Set up file system observer
    observer = Observer()
    handler = LogFileHandler(agent)
    watch_path = Path(watch_path)
    observer.schedule(handler, str(watch_path), recursive=False)
    observer.start()

    try:
        # Process existing files first
        for file in watch_path.glob("*.csv"):
            await agent.process_file(file)

        # Keep the observer running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        agent.logger.info("Agent stopped by user")
    finally:
        observer.join()


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing.

    Returns:
        ArgumentParser instance configured with required arguments
    """
    parser = argparse.ArgumentParser(description="Run an agent to monitor and analyze log files for security threats")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--watch", type=str, required=True, help="Path to monitor for log files")
    parser.add_argument("--output", type=str, help="Directory to save analysis results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing (default: 8)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for threat detection (default: 0.7)",
    )
    return parser


def main() -> None:
    """Main entry point for the agent script."""
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        asyncio.run(
            run_agent(
                model_path=args.model,
                watch_path=args.watch,
                batch_size=args.batch_size,
                threshold=args.threshold,
                output_dir=args.output,
            )
        )
    except Exception as e:
        print(f"Error running agent: {e!s}")


if __name__ == "__main__":
    main()
