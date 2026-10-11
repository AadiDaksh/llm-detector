"""Tokenizer and batches using PyTorch"""
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<PAD>"  # Padding at the end of a prompt or response.
UNK_TOKEN = "<UNK>"  # Used when a word is not in the vocabulary.
SEP_TOKEN = "<SEP>"  # Separates the prompt and response in "both" mode.


def tokenize(text):
    if pd.isna(text):
        return []

    return re.findall(r"\w+|[^\w\s]", str(text).lower())


def get_tokens(row, mode):
    if mode == "input":
        return tokenize(row["input"])

    if mode == "output":
        return tokenize(row["output"])

    if mode == "both":
        return tokenize(row["input"]) + [SEP_TOKEN] + tokenize(row["output"])

    raise ValueError("Mode must be 'input', 'output', or 'both'.")


def build_vocab(texts, max_vocab_size=20000, min_freq=2):
    counter = Counter()

    for tokens in texts:
        counter.update(tokens)

    vocab = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
        SEP_TOKEN: 2
    }

    for token, count in counter.most_common():
        if count < min_freq:
            break

        if token not in vocab and len(vocab) < max_vocab_size:
            vocab[token] = len(vocab)

    return vocab


class LLMDataset(Dataset):
    def __init__(self, dataframe, vocab, mode="output", max_length=256):
        if mode not in {"input", "output", "both"}:
            raise ValueError("Mode must be 'input', 'output', or 'both'.")

        if max_length < 1:
            raise ValueError("Maximum length must be at least 1.")

        if mode == "both" and max_length < 2:
            raise ValueError("Maximum length must be at least 2 for 'both' mode.")

        self.vocab = vocab
        self.mode = mode
        self.max_length = max_length
        self.samples = []

        for _, row in dataframe.iterrows():
            tokens = get_tokens(row, mode)

            if mode == "both":
                question = tokenize(row["input"])
                response = tokenize(row["output"])

                # Reserve space for the question, response, and separator.
                question_budget = min(len(question), max(1, max_length // 4))
                response_budget = max_length - question_budget - 1

                tokens = (
                    question[:question_budget]
                    + [SEP_TOKEN]
                    + response[:response_budget]
                )

            # Convert tokens into numerical IDs.
            ids = [
                vocab.get(token, vocab[UNK_TOKEN])
                for token in tokens[:max_length]
            ]

            # Pad shorter sequences to the maximum length.
            ids += [vocab[PAD_TOKEN]] * (max_length - len(ids))

            self.samples.append(
                (torch.tensor(ids, dtype=torch.long), int(row["label"]))
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        ids, label = self.samples[index]

        return ids, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    data_dir="data/processed",
    mode="output",
    batch_size=32,
    max_length=256,
    max_vocab_size=20000,
    min_freq=2,
    num_workers=0
):
    """Create training, validation, and testing data loaders."""

    if mode not in {"input", "output", "both"}:
        raise ValueError("Mode must be 'input', 'output', or 'both'.")

    data_dir = Path(data_dir)
    frames = {}

    # Load the processed datasets.
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run python prepare_data.py first."
            )

        df = pd.read_csv(path).fillna({
            "input": "",
            "output": ""
        })

        required = {"input", "output", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} is missing columns: {required - set(df.columns)}"
            )

        frames[split] = df

    # Build the vocabulary using only training data.
    train_tokens = (
        get_tokens(row, mode)
        for _, row in frames["train"].iterrows()
    )

    vocab = build_vocab(
        train_tokens,
        max_vocab_size=max_vocab_size,
        min_freq=min_freq
    )

    # Convert the dataframes into PyTorch datasets.
    datasets = {
        split: LLMDataset(
            frame,
            vocab,
            mode=mode,
            max_length=max_length
        )
        for split, frame in frames.items()
    }

    # Create batches for training, validation, and testing.
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers
        )
        for split, dataset in datasets.items()
    }

    return (
        loaders["train"],
        loaders["val"],
        loaders["test"],
        vocab
    )


if __name__ == "__main__":
    for mode in ("input", "output", "both"):
        train_loader, val_loader, test_loader, vocab = create_dataloaders(
            mode=mode
        )

        input_ids, labels = next(iter(train_loader))

        print(f"\nMode: {mode}")
        print(
            f"Train / val / test: {len(train_loader.dataset)} / "
            f"{len(val_loader.dataset)} / {len(test_loader.dataset)}"
        )
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Batch shape: {tuple(input_ids.shape)}")
        print(f"Labels shape: {tuple(labels.shape)}")
        print(f"First label: {labels[0].item()}")
        print(f"First 20 token IDs: {input_ids[0, :20].tolist()}")

    
