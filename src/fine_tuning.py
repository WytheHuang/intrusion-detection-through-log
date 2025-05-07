"""Fine-tuning module for the log analysis model.

This module handles the fine-tuning process of the LLM for log analysis,
including training loop, validation, and model checkpointing.
"""

import json
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier


class ModelTrainer:
    """Trainer class for fine-tuning the log analysis model."""

    def __init__(
        self,
        model_name: str = "gemma:12b",
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        device: str | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model_name: Name of the base model to fine-tune
            num_labels: Number of classification labels
            learning_rate: Learning rate for training
            device: Device to use for training (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model and move to device
        self.model = LogAnalysisModel(model_name=model_name, num_labels=num_labels)
        self.classifier = LogClassifier(self.model, device=self.device)

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.classifier.classifier.parameters(),
            lr=learning_rate,
        )
        self.scheduler = None  # Will be initialized during training

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> float:
        """Train for one epoch.

        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number

        Returns:
            Average loss for the epoch
        """
        self.classifier.classifier.train()
        total_loss = 0
        num_batches = len(train_loader)

        progress_bar = tqdm(
            train_loader,
            desc=f"Training Epoch {epoch}",
            leave=False,
        )

        for batch in progress_bar:
            loss = self.classifier.train_step(batch)
            total_loss += loss

            if self.scheduler is not None:
                self.scheduler.step()

            progress_bar.set_postfix({"loss": f"{loss:.4f}"})

        return total_loss / num_batches

    def validate(self, val_loader: torch.utils.data.DataLoader) -> dict[str, float]:
        """Evaluate the model on validation data.

        Args:
            val_loader: DataLoader for validation data

        Returns:
            Dictionary containing validation metrics
        """
        self.classifier.classifier.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                result = self.classifier.evaluate(batch)
                total_loss += result["loss"]
                all_preds.extend(result["predictions"].numpy())
                all_labels.extend(result["labels"].numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = sum(preds == labels for preds, labels in zip(all_preds, all_labels)) / len(all_preds)

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
        }

    def train(
        self,
        train_data: str,
        num_epochs: int = 10,
        batch_size: int = 16,
        checkpoint_dir: str | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            train_data: Path to training data CSV file
            num_epochs: Number of epochs to train
            batch_size: Batch size for training
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary containing training history
        """
        # Create checkpoint directory if needed
        if checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Prepare data
        processor = LogProcessor(batch_size=batch_size)
        train_loader, val_loader, _ = processor.prepare_data(train_data)

        # Setup learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * num_epochs,
        )

        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            history["train_loss"].append(train_loss)

            # Validate
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_accuracy"].append(val_metrics["val_accuracy"])

            print(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Accuracy: {val_metrics['val_accuracy']:.4f}"
            )

            # Save checkpoint if validation loss improved
            if checkpoint_dir and val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pt"
                self.save_checkpoint(checkpoint_file, epoch, history)
                print(f"Saved checkpoint to {checkpoint_file}")

        return history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        history: dict[str, list[float]],
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            history: Training history dictionary
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "classifier_state_dict": self.classifier.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "history": history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.classifier.classifier.load_state_dict(checkpoint["classifier_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])


def main() -> None:
    """Run the fine-tuning process."""
    # Initialize trainer
    trainer = ModelTrainer()

    # Set paths
    data_path = "data/synthetic_logs.csv"
    checkpoint_dir = "checkpoints"

    # Train the model
    print("Starting training...")
    history = trainer.train(
        train_data=data_path,
        num_epochs=5,
        batch_size=16,
        checkpoint_dir=checkpoint_dir,
    )

    # Save training history
    history_file = Path(checkpoint_dir) / "training_history.json"
    with history_file.open("w") as f:
        json.dump(history, f, indent=2)

    print(f"Training complete! History saved to {history_file}")


if __name__ == "__main__":
    main()
