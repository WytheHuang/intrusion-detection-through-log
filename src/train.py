import argparse
import os

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from data_processor import LogProcessor
from model import LogAnalysisModel, LogClassifier


def train(args):
    # Initialize data processor
    processor = LogProcessor(batch_size=args.batch_size)

    # Load and prepare data
    train_loader, val_loader, test_loader = processor.prepare_data(
        args.data_path, label_column=args.label_column, text_column=args.text_column
    )

    # Initialize model and classifier
    model = LogAnalysisModel(model_name=args.model_name, num_labels=args.num_labels)
    classifier = LogClassifier(model)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_losses = []
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch in train_loop:
            loss = classifier.train_step(batch)
            train_losses.append(loss)
            train_loop.set_postfix({"loss": np.mean(train_losses)})

        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]")
        for batch in val_loop:
            result = classifier.evaluate(batch)
            val_losses.append(result["loss"])
            val_preds.extend(result["predictions"].numpy())
            val_labels.extend(result["labels"].numpy())
            val_loop.set_postfix({"loss": np.mean(val_losses)})

        # Save best model
        val_loss = np.mean(val_losses)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)
            model.save(os.path.join(args.output_dir, "best_model.pt"))

        # Print validation metrics
        print("\nValidation Report:")
        print(classification_report(val_labels, val_preds))

    # Final test evaluation
    model.load(os.path.join(args.output_dir, "best_model.pt"))
    test_preds = []
    test_labels = []
    test_loop = tqdm(test_loader, desc="Testing")
    for batch in test_loop:
        result = classifier.evaluate(batch)
        test_preds.extend(result["predictions"].numpy())
        test_labels.extend(result["labels"].numpy())

    print("\nTest Report:")
    print(classification_report(test_labels, test_preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the log data CSV file")
    parser.add_argument("--label_column", type=str, default="label", help="Name of the label column")
    parser.add_argument("--text_column", type=str, default="log_text", help="Name of the text column")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of classes")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save the model")
    parser.add_argument("--model_name", type=str, default="gemma3:12b", help="Name of the Ollama model to use")

    args = parser.parse_args()
    train(args)
