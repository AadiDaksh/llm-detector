
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from torch.utils.data import DataLoader

from data_pipeline import LLMDataset
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier


LABEL_NAMES = ["GPT", "Claude", "Gemini"]


def create_model(model_name, vocab_size):
    """Initialize the CNN or LSTM classifier."""
    if model_name == "cnn":
        return CNNClassifier(
            vocab_size=vocab_size
        )

    if model_name == "lstm":
        return LSTMClassifier(
            vocab_size=vocab_size
        )

    raise ValueError(f"Unsupported model: {model_name}")


def load_model(checkpoint_path, model_name, mode, device):
    """Load the trained model and its saved vocabulary."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    if checkpoint["model"] != model_name:
        raise ValueError("Checkpoint model does not match selected model.")

    if checkpoint["mode"] != mode:
        raise ValueError("Checkpoint mode does not match selected mode.")

    vocab = checkpoint["vocab"]

    model = create_model(
        model_name=model_name,
        vocab_size=len(vocab)
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, checkpoint


def load_test_data(vocab, mode, batch_size):
    """Load the test dataset using the training vocabulary."""
    test_path = Path("data/processed/test.csv")

    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing {test_path}. Run prepare_data.py first."
        )

    test_df = pd.read_csv(test_path).fillna({
        "input": "",
        "output": ""
    })

    test_dataset = LLMDataset(
        dataframe=test_df,
        vocab=vocab,
        mode=mode,
        max_length=256
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


def evaluate_model(model, test_loader, device):
    """Generate predictions on unseen test examples."""
    predictions = []
    true_labels = []

    model.eval()

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)

            logits = model(input_ids)
            predicted = logits.argmax(dim=1)

            predictions.extend(predicted.cpu().tolist())
            true_labels.extend(labels.tolist())

    return np.array(true_labels), np.array(predictions)


def save_results(true_labels, predictions, model_name, mode):
    """Calculate metrics and save evaluation results."""
    save_dir = Path("results") / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    accuracy = accuracy_score(true_labels, predictions)

    report = classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2]
    )

    print(f"\nTest Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    # Save evaluation metrics as JSON.
    metrics = {
        "model": model_name,
        "mode": mode,
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }

    metrics_path = save_dir / f"{mode}_metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Generate and save the confusion matrix.
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABEL_NAMES
    )

    display.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.title(f"{model_name.upper()} Confusion Matrix ({mode})")
    plt.tight_layout()

    figure_path = save_dir / f"{mode}_confusion_matrix.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()

    # Save individual predictions for later analysis.
    predictions_df = pd.DataFrame({
        "true_label": true_labels,
        "predicted_label": predictions,
        "correct": true_labels == predictions
    })

    predictions_path = save_dir / f"{mode}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Confusion matrix saved to: {figure_path}")
    print(f"Predictions saved to: {predictions_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LLM classifier."
    )

    parser.add_argument(
        "--model",
        choices=["cnn", "lstm"],
        default="cnn"
    )

    parser.add_argument(
        "--mode",
        choices=["input", "output", "both"],
        default="output"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        Path("results") / args.model / f"best_{args.mode}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run train.py first."
        )

    # Load the best trained model.
    model, vocab, checkpoint = load_model(
        checkpoint_path,
        args.model,
        args.mode,
        device
    )

    # Prepare unseen test examples.
    test_loader = load_test_data(
        vocab,
        args.mode,
        args.batch_size
    )

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Test examples: {len(test_loader.dataset)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Best training epoch: {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_accuracy']:.2%}")

    # Evaluate the trained model.
    true_labels, predictions = evaluate_model(
        model,
        test_loader,
        device
    )

    save_results(
        true_labels,
        predictions,
        args.model,
        args.mode
    )


if __name__ == "__main__":
    main()
