
"""RQ3: Task-held-out LLM attribution with standalone CNN and BiLSTM."""

import argparse
import json
import random
import re
import sys

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import Dataset, DataLoader


SEED = 42

RULES = {
    "coding": (
        r"\b(code|coding|program|programming|python|java|javascript|"
        r"function|algorithm|data structure|binary search|linked list|"
        r"stack|queue|heap|tree|graph|sorting|sort|recursion|compiler|"
        r"database|sql|debug|api|operating system|thread|process|"
        r"memory|complexity|runtime|computer science|hash table|"
        r"pointer|class|object-oriented|software|network|linux|"
        r"javascript)\b"
    ),
    "math": (
        r"\b(calculate|equation|integral|derivative|probability|"
        r"statistics|algebra|geometry|trigonometry|matrix|matrices|"
        r"polynomial|factorial|theorem|proof|solve for|mathematical|"
        r"arithmetic|fraction|quadratic|calculus)\b"
    ),
    "creative": (
        r"\b(write a (story|poem|song|script|dialogue|fiction)|"
        r"short story|creative writing|fictional|narrative|haiku|"
        r"poetry|poem|fairy tale|screenplay|character development|"
        r"write a letter|compose a)\b"
    ),
    "factual": (
        r"\b(who (was|is|invented|discovered)|when (was|did)|"
        r"where (is|was)|what (is|are|was|were)|history of|"
        r"historical|capital of|explain|describe|difference between|"
        r"how does|why does|definition of)\b"
    ),
}

TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.I)


def task(text):
    text = str(text).lower()

    # Require an unambiguous specialized category.
    matches = [
        category
        for category in ("creative", "math", "coding")
        if re.search(RULES[category], text)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return "ambiguous"

    if re.search(RULES["factual"], text):
        return "factual"

    return "unclassified"


def read_csv(path):
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="latin-1",
            low_memory=False,
        )


def detect_col(columns, options):
    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for name in options:
        if name in lookup:
            return lookup[name]

    return None


def prepare(args, out):
    df = read_csv(args.csv)

    q = detect_col(
        df.columns,
        ["question", "llm_input", "input", "prompt", "instruction", "user_prompt"],
    )

    a = detect_col(
        df.columns,
        ["answer", "llm_output", "output", "response", "generated_text", "text"],
    )

    s = detect_col(
        df.columns,
        ["source", "llm_name", "model", "model_name", "label"],
    )

    if not all([q, a, s]):
        raise ValueError(
            f"Could not identify Question/Answer/Source columns. "
            f"Found: {list(df.columns)}"
        )

    df = df[[q, a, s]].copy()
    df.columns = ["prompt", "response", "source"]

    df = df.dropna().astype(str)

    df["prompt"] = df.prompt.str.strip()
    df["response"] = df.response.str.strip()
    df["source"] = df.source.str.strip()

    df = df[
        (df.prompt.str.len() > 5)
        & (df.response.str.len() > 20)
    ]

    print(
        "Available model labels:",
        df.source.value_counts().head(30).to_dict(),
    )

    if args.labels:
        labels = [
            label.strip()
            for label in args.labels.split(",")
        ]
        df = df[df.source.isin(labels)]

    else:
        labels = df.source.value_counts().head(3).index.tolist()
        df = df[df.source.isin(labels)]
        print("Automatically selected labels:", labels)

    if len(labels) != 3 or df.source.nunique() != 3:
        raise ValueError(
            "RQ3 requires three distinct model labels with data. "
            'Use --labels "EXACT1,EXACT2,EXACT3"'
        )

    df["task"] = df.prompt.map(task)

    df.groupby(["task", "source"]).size().unstack(
        fill_value=0
    ).to_csv(out / "task_counts.csv")

    print(
        "\nTask counts by model:\n",
        pd.crosstab(df.task, df.source).to_string(),
    )

    df[["prompt", "source", "task"]].sample(
        min(80, len(df)),
        random_state=SEED,
    ).to_csv(
        out / "task_audit_sample.csv",
        index=False,
    )

    train_tasks = [
        value.strip()
        for value in args.train_tasks.split(",")
    ]

    test_task = args.test_task

    if test_task in train_tasks:
        raise ValueError(
            "Test task must not be a training task"
        )

    known = set(RULES)

    if set(train_tasks + [test_task]) - known:
        raise ValueError(
            f"Valid categories: {sorted(known)}"
        )

    train_pool = df[
        df.task.isin(train_tasks)
    ].copy()

    test_pool = df[
        df.task == test_task
    ].copy()

    if train_pool.empty or test_pool.empty:
        raise ValueError(
            "No examples for selected task categories. "
            "See task_counts.csv; choose --test-task or --train-tasks."
        )

    # Balance sources in training and testing.
    rng = np.random.RandomState(SEED)

    train_parts = []
    test_parts = []

    for source in labels:
        tr = train_pool[
            train_pool.source == source
        ]

        te = test_pool[
            test_pool.source == source
        ]

        if (
            len(tr) < args.min_per_class
            or len(te) < args.min_test_per_class
        ):
            raise ValueError(
                f"Insufficient {source}: "
                f"training={len(tr)}, unseen test={len(te)}; "
                f"need >= {args.min_per_class} and "
                f"{args.min_test_per_class}. "
                "Check rq3/results/task_counts.csv. "
                "You may choose a different --test-task, "
                "but do not invent results."
            )

        train_parts.append(
            tr.sample(
                n=min(len(tr), args.max_train_per_class),
                random_state=SEED,
            )
        )

        test_parts.append(
            te.sample(
                n=min(len(te), args.max_test_per_class),
                random_state=SEED,
            )
        )

    train_pool = pd.concat(
        train_parts
    ).reset_index(drop=True)

    test_df = pd.concat(
        test_parts
    ).sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # Split validation by unique prompts.
    prompts = np.array(
        sorted(train_pool.prompt.unique())
    )

    rng.shuffle(prompts)

    nval = max(
        1,
        int(round(0.2 * len(prompts))),
    )

    val_prompts = set(prompts[:nval])

    train_df = train_pool[
        ~train_pool.prompt.isin(val_prompts)
    ].copy()

    val_df = train_pool[
        train_pool.prompt.isin(val_prompts)
    ].copy()

    if (
        train_df.source.nunique() != 3
        or val_df.source.nunique() != 3
    ):
        raise ValueError(
            "Validation split lacks one or more model classes; "
            "increase data or change train tasks."
        )

    for name, split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        split[
            ["prompt", "response", "source", "task"]
        ].to_csv(
            out / f"{name}_split.csv",
            index=False,
        )

        print(
            name,
            len(split),
            split.source.value_counts().to_dict(),
        )

    print(
        "Task leakage check: train tasks=",
        train_tasks,
        "test task=",
        test_task,
    )

    print(
        "Prompt overlap:",
        len(
            set(train_df.prompt)
            & set(val_df.prompt)
        ),
        len(
            set(train_pool.prompt)
            & set(test_df.prompt)
        ),
    )

    return train_df, val_df, test_df, labels


class TextSet(Dataset):
    def __init__(self, df, vocab, labels, max_len):
        self.x = []
        self.y = []

        for row in df.itertuples():
            tokens = TOKEN.findall(
                row.response.lower()
            )[:max_len]

            ids = [
                vocab.get(token, 1)
                for token in tokens
            ]

            padded = ids + [0] * (
                max_len - len(ids)
            )

            self.x.append(padded)
            self.y.append(
                labels.index(row.source)
            )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(
            self.x[index],
            dtype=torch.long,
        )

        y = torch.tensor(
            self.y[index],
            dtype=torch.long,
        )

        return x, y


# --------------------------------------------------
# CNN architecture
# --------------------------------------------------

class CNN(nn.Module):
    def __init__(self, v, dim=128, filters=100):
        super().__init__()

        self.emb = nn.Embedding(
            v,
            dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(dim, filters, kernel)
            for kernel in (3, 4, 5)
        ])

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(filters * 3, 3)

    def forward(self, x):
        z = self.emb(x).transpose(1, 2)

        z = torch.cat(
            [
                torch.relu(conv(z)).amax(dim=2)
                for conv in self.convs
            ],
            dim=1,
        )

        return self.fc(
            self.dropout(z)
        )


class BiLSTM(nn.Module):
    def __init__(self, v, dim=128, hidden=128):
        super().__init__()

        self.emb = nn.Embedding(
            v,
            dim,
            padding_idx=0,
        )

        self.rnn = nn.LSTM(
            dim,
            hidden,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden * 2, 3)

    def forward(self, x):
        z = self.emb(x)

        lengths = x.ne(0).sum(1).clamp(
            min=1
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            z,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h, _) = self.rnn(packed)

        combined = torch.cat(
            [h[-2], h[-1]],
            dim=1,
        )

        return self.fc(
            self.dropout(combined)
        )


def evaluate(model, loader, device):
    model.eval()

    ys = []
    pred = []
    total_loss = 0

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            total_loss += (
                loss_fn(logits, y).item()
                * len(y)
            )

            ys.extend(
                y.cpu().tolist()
            )

            pred.extend(
                logits.argmax(1).cpu().tolist()
            )

    metrics = {
        "accuracy": accuracy_score(ys, pred),
        "macro_f1": f1_score(
            ys,
            pred,
            average="macro",
            zero_division=0,
        ),
        "loss": total_loss / len(ys),
    }

    return metrics, ys, pred


def run(name, train, val, test, labels, args, out):
    words = Counter(
        token
        for response in train.response
        for token in TOKEN.findall(
            response.lower()
        )
    )

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, count in words.most_common(
        args.vocab_size - 2
    ):
        if count >= 2:
            vocab[token] = len(vocab)

    (out / f"{name}_vocab.json").write_text(
        json.dumps(vocab),
        encoding="utf-8",
    )

    tr = DataLoader(
        TextSet(
            train,
            vocab,
            labels,
            args.max_len,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    va = DataLoader(
        TextSet(
            val,
            vocab,
            labels,
            args.max_len,
        ),
        batch_size=args.batch_size,
    )

    te = DataLoader(
        TextSet(
            test,
            vocab,
            labels,
            args.max_len,
        ),
        batch_size=args.batch_size,
    )

    if name == "cnn":
        model = CNN(len(vocab))
    else:
        model = BiLSTM(len(vocab))

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    loss_fn = nn.CrossEntropyLoss()

    history = []
    best = -1
    best_epoch = 0

    print(
        f"\n{name.upper()} | "
        f"device={device} | "
        f"vocab={len(vocab)}"
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        total = 0
        correct = 0
        loss_sum = 0

        for x, y in tr:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            total += len(y)

            correct += (
                logits.argmax(1) == y
            ).sum().item()

            loss_sum += (
                loss.item() * len(y)
            )

        vm, _, _ = evaluate(
            model,
            va,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / total,
            "train_accuracy": correct / total,
            "val_loss": vm["loss"],
            "val_accuracy": vm["accuracy"],
        })

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train={correct / total:.2%} "
            f"val={vm['accuracy']:.2%}"
        )

        if vm["accuracy"] > best:
            best = vm["accuracy"]
            best_epoch = epoch

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocab": vocab,
                    "labels": labels,
                    "epoch": epoch,
                    "max_len": args.max_len,
                    "architecture": name,
                },
                out / f"{name}_best.pt",
            )

    pd.DataFrame(history).to_csv(
        out / f"{name}_training_history.csv",
        index=False,
    )

    state = torch.load(
        out / f"{name}_best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        state["state_dict"]
    )

    tm, ys, pred = evaluate(
        model,
        te,
        device,
    )

    report = classification_report(
        ys,
        pred,
        labels=[0, 1, 2],
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        ys,
        pred,
        labels=[0, 1, 2],
    )

    # Save confusion matrix visualization.
    fig, ax = plt.subplots(
        figsize=(5, 4)
    )

    im = ax.imshow(
        cm,
        cmap="Blues",
    )

    fig.colorbar(im, ax=ax)

    ax.set_xticks(
        range(3),
        labels=labels,
    )

    ax.set_yticks(
        range(3),
        labels=labels,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(
        f"RQ3 {name.upper()} "
        f"unseen task: {args.test_task}"
    )

    for i in range(3):
        for j in range(3):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        out / f"{name}_confusion_matrix.png",
        dpi=160,
    )

    plt.close(fig)

    # Save predictions.
    pd.DataFrame({
        "actual": [
            labels[i]
            for i in ys
        ],
        "predicted": [
            labels[i]
            for i in pred
        ],
    }).to_csv(
        out / f"{name}_predictions.csv",
        index=False,
    )

    # Save evaluation metrics.
    metrics = {
        "model": name,
        "test_task": args.test_task,
        "train_tasks": args.train_tasks.split(","),
        "test_examples": len(ys),
        "best_epoch": best_epoch,
        "best_val_accuracy": best,
        "test_accuracy": tm["accuracy"],
        "test_macro_f1": tm["macro_f1"],
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    (out / f"{name}_metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{name.upper()} UNSEEN {args.test_task}: "
        f"accuracy={tm['accuracy']:.2%}, "
        f"macro F1={tm['macro_f1']:.3f}"
    )

    print(
        classification_report(
            ys,
            pred,
            labels=[0, 1, 2],
            target_names=labels,
            zero_division=0,
        )
    )

    return metrics


# --------------------------------------------------
# Command-line interface
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=["cnn", "lstm", "all"],
        default="all",
    )

    parser.add_argument(
        "--labels",
        help=(
            "Three exact model labels, "
            "comma-separated; default top three"
        ),
    )

    parser.add_argument(
        "--train-tasks",
        default="coding,math,factual",
    )

    parser.add_argument(
        "--test-task",
        default="creative",
        choices=list(RULES),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--max-test-per-class",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--min-per-class",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--min-test-per-class",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    out = (
        Path(__file__).resolve().parent
        / "results"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, val, test, labels = prepare(
        args,
        out,
    )

    all_metrics = {}

    model_names = (
        ["cnn", "lstm"]
        if args.model == "all"
        else [args.model]
    )

    for name in model_names:
        all_metrics[name] = run(
            name,
            train,
            val,
            test,
            labels,
            args,
            out,
        )

    (out / "summary.json").write_text(
        json.dumps(
            all_metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nDONE. Results:",
        out,
    )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print(
            f"\nRQ3 STOPPED: {error}",
            file=sys.stderr,
        )
        sys.exit(1)
