
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_pipeline import create_dataloaders
from models.cnn import CNNClassifier
from models.lstm import LSTMClassifier


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Create the selected model.
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


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train or validate the model for one epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(input_ids)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(
        description="Train the LLM classification model."
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

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, and lr must be positive")

    # Set random seeds and select CPU or GPU.
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load training and validation datasets.
    train_loader, val_loader, _, vocab = create_dataloaders(
        mode=args.mode,
        batch_size=args.batch_size
    )

    # Initialize either CNN or LSTM
    model = create_model(
        model_name=args.model,
        vocab_size=len(vocab)
    ).to(device)

    # Define the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # Create the directory for saved models.
    save_dir = Path("results") / args.model
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{args.mode}_best.pt"
    best_val_accuracy = -1.0

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training examples: {len(train_loader.dataset)}")
    print(f"Validation examples: {len(val_loader.dataset)}")

    # Train and validate for the requested number of epochs.
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"\nEpoch {epoch}/{args.epochs}"
            f"\nTrain Loss: {train_loss:.4f}"
            f" | Train Accuracy: {train_acc:.2%}"
            f"\nVal Loss: {val_loss:.4f}"
            f" | Val Accuracy: {val_acc:.2%}"
        )

        # Save the model whenever validation accuracy improves.
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab": vocab,
                    "mode": args.mode,
                    "model": args.model,
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                },
                save_path
            )

            print(f"Saved best model to: {save_path}")

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2%}")
    print(f"Model saved at: {save_path}")


if __name__ == "__main__":
    main()
