"""RQ3 — Do LLM fingerprints generalize across tasks or domains?

TXD-22 is entirely Q&A; there is no coding, writing, or math task mix.
The feasible generalization axis is topic domain. We split the prompts
into STEM, NON-STEM, and OTHER using keyword matching, then:

    1. Train an LSTM on STEM responses only.
    2. Evaluate on held-out STEM responses (in-domain).
    3. Evaluate on NON-STEM responses (cross-domain).

If accuracy stays high cross-domain, the model is picking up general LLM
style; if it drops sharply, it was keying on topic/vocabulary signals.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import LLMDataset, build_vocab, get_tokens
from models.lstm import LSTMClassifier


SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
MODE = "output"

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/rq3")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


STEM_KEYWORDS = [
    "nuclear", "reactor", "science", "scientific", "technology", "computer",
    "software", "hardware", "math", "physics", "biology", "chemistry",
    "engineering", "mechanical", "electrical", "quantum", "dna", "cell",
    "organism", "molecule", "algorithm", "data", "nano", "wave", "voltage",
    "current", "circuit", "function", "equation", "formula", "theorem",
    "signal", "frequency", "thyristor", "capacitor", "resistor",
    "semiconductor", "oxide", "polymer", "enzyme", "bacteria", "virus",
    "animal", "plant", "ecology", "ecosystem", "climate", "weather",
    "geology", "atom", "electron", "proton", "neutron", "radiation",
    "energy", "power", "machine", "automation", "robot", "artificial",
    "neural", "network", "control system", "lens", "optics", "disaster",
    "earthquake", "volcanic", "pharmac", "medic", "disease", "treatment",
    "trajectory", "navigation", "sensor", "probability", "statistic",
    "calculation", "measure", "numerical", "programming", "code",
    "database", "cyber",
]

NONSTEM_KEYWORDS = [
    "social", "society", "political", "politics", "gender", "criminology",
    "sociology", "history", "historical", "culture", "cultural", "religion",
    "religious", "media", "communication", "communicat", "education",
    "psychology", "liberty", "freedom", "philosophy", "ethic", "renaissance",
    "colonial", "nationalism", "globalization", "literature", "literary",
    "poem", "poetry", "art", "artistic", "museum", "music", "theatre",
    "theater", "film", "cinema", "journalism", "advertising", "audience",
    "legal", "law", "justice", "crime", "prison", "constitution",
    "parliament", "democracy", "government", "economy", "economic",
    "finance", "market", "business", "management", "marketing", "trade",
    "tourism", "sport", "language", "linguistic", "grammar", "homonym",
    "homophone", "stereotype", "race", "feminis", "patriarch", "identity",
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def classify_domain(prompt):
    p = str(prompt).lower()
    stem = any(k in p for k in STEM_KEYWORDS)
    non = any(k in p for k in NONSTEM_KEYWORDS)
    if stem and not non:
        return "STEM"
    if non and not stem:
        return "NONSTEM"
    if stem and non:
        return "BOTH"
    return "OTHER"


def load_all_examples():
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(PROCESSED_DIR / f"{split}.csv").fillna(
            {"input": "", "output": ""}
        )
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["domain"] = df["input"].apply(classify_domain)
    return df


def build_cross_domain_splits(df):
    # Train/val from STEM, test from both STEM (in-domain) and NON-STEM
    # (cross-domain). Split by prompt so no prompt crosses splits.
    stem = df[df["domain"] == "STEM"].copy()
    nonstem = df[df["domain"] == "NONSTEM"].copy()

    rng = np.random.default_rng(SEED)
    stem_prompts = stem["input"].unique().tolist()
    rng.shuffle(stem_prompts)

    n_train = int(0.80 * len(stem_prompts))
    n_val = int(0.10 * len(stem_prompts))

    train_prompts = set(stem_prompts[:n_train])
    val_prompts = set(stem_prompts[n_train : n_train + n_val])
    stem_test_prompts = set(stem_prompts[n_train + n_val :])

    train_df = stem[stem["input"].isin(train_prompts)].reset_index(drop=True)
    val_df = stem[stem["input"].isin(val_prompts)].reset_index(drop=True)
    stem_test_df = stem[stem["input"].isin(stem_test_prompts)].reset_index(
        drop=True
    )
    nonstem_test_df = nonstem.reset_index(drop=True)

    return train_df, val_df, stem_test_df, nonstem_test_df


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
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
        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += n
    return total_loss / total, correct / total


def evaluate(model, loader, device, label_names):
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
    )
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            logits = model(input_ids)
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            trues.extend(labels.tolist())
    acc = accuracy_score(trues, preds)
    macro_f1 = f1_score(trues, preds, average="macro")
    per_class_f1 = f1_score(trues, preds, average=None).tolist()
    cm = confusion_matrix(trues, preds, labels=list(range(len(label_names))))
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class_f1": {label_names[i]: float(per_class_f1[i])
                         for i in range(len(label_names))},
        "confusion_matrix": cm.tolist(),
        "n": int(len(trues)),
    }


def main():
    set_seed()

    with open(PROCESSED_DIR / "label_mapping.json") as f:
        label_map = json.load(f)
    id_to_name = {v: k for k, v in label_map.items()}
    label_names = [id_to_name[i] for i in range(len(id_to_name))]

    df = load_all_examples()
    print("Domain distribution (per LLM):")
    print(pd.crosstab(df["model"], df["domain"]))

    train_df, val_df, stem_test_df, nonstem_test_df = build_cross_domain_splits(df)

    print(f"\nTrain (STEM): {len(train_df)}  Val (STEM): {len(val_df)}")
    print(f"Test STEM (in-domain):     {len(stem_test_df)}")
    print(f"Test NON-STEM (cross):     {len(nonstem_test_df)}")
    print()
    print("Train class balance:")
    print(train_df["model"].value_counts().to_string())
    print()
    print("STEM test class balance:")
    print(stem_test_df["model"].value_counts().to_string())
    print()
    print("NON-STEM test class balance:")
    print(nonstem_test_df["model"].value_counts().to_string())

    # Build vocab on training texts only.
    train_tokens = (get_tokens(row, MODE) for _, row in train_df.iterrows())
    vocab = build_vocab(train_tokens, max_vocab_size=20000, min_freq=2)
    print(f"\nVocab size (STEM-trained): {len(vocab)}")

    train_ds = LLMDataset(train_df, vocab, mode=MODE, max_length=MAX_LENGTH)
    val_ds = LLMDataset(val_df, vocab, mode=MODE, max_length=MAX_LENGTH)
    stem_test_ds = LLMDataset(
        stem_test_df, vocab, mode=MODE, max_length=MAX_LENGTH
    )
    nonstem_test_ds = LLMDataset(
        nonstem_test_df, vocab, mode=MODE, max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    stem_test_loader = DataLoader(
        stem_test_ds, batch_size=BATCH_SIZE, shuffle=False
    )
    nonstem_test_loader = DataLoader(
        nonstem_test_ds, batch_size=BATCH_SIZE, shuffle=False
    )

    device = torch.device("cpu")
    model = LSTMClassifier(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val, best_state = -1.0, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(
            f"Epoch {epoch}/{EPOCHS}  "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.2%} | "
            f"val_loss={vl_loss:.4f} val_acc={vl_acc:.2%}  ({dt:.1f}s)"
        )
        history.append(
            {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
             "val_loss": vl_loss, "val_acc": vl_acc}
        )
        if vl_acc > best_val:
            best_val = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\nBest val accuracy: {best_val:.2%}")
    model.load_state_dict(best_state)

    stem_eval = evaluate(model, stem_test_loader, device, label_names)
    nonstem_eval = evaluate(model, nonstem_test_loader, device, label_names)

    print("\n=== STEM test (in-domain) ===")
    print(f"accuracy = {stem_eval['accuracy']:.4f}  "
          f"macroF1 = {stem_eval['macro_f1']:.4f}  n={stem_eval['n']}")
    for name, f1 in stem_eval["per_class_f1"].items():
        print(f"  {name}: F1={f1:.4f}")
    print("confusion (rows=true, cols=pred):")
    for row in stem_eval["confusion_matrix"]:
        print(" ", row)

    print("\n=== NON-STEM test (cross-domain) ===")
    print(f"accuracy = {nonstem_eval['accuracy']:.4f}  "
          f"macroF1 = {nonstem_eval['macro_f1']:.4f}  n={nonstem_eval['n']}")
    for name, f1 in nonstem_eval["per_class_f1"].items():
        print(f"  {name}: F1={f1:.4f}")
    print("confusion (rows=true, cols=pred):")
    for row in nonstem_eval["confusion_matrix"]:
        print(" ", row)

    out = {
        "setup": {
            "model": "lstm",
            "mode": MODE,
            "trained_on": "STEM responses only",
            "vocab_size": len(vocab),
            "train_n": len(train_df),
            "val_n": len(val_df),
            "stem_test_n": len(stem_test_df),
            "nonstem_test_n": len(nonstem_test_df),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "max_length": MAX_LENGTH,
            "best_val_accuracy": best_val,
        },
        "history": history,
        "in_domain_stem_test": stem_eval,
        "cross_domain_nonstem_test": nonstem_eval,
        "label_names": label_names,
    }

    out_path = RESULTS_DIR / "lstm_stem_to_nonstem.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
