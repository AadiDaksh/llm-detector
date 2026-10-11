
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SEED = 42
SAMPLES_PER_CLASS = 2000

LABELS = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2,
}


def load_data():
    csv_files = sorted(RAW_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV file found in {RAW_DIR}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            "Multiple CSV files found in data/raw. "
            "Keep only the TXD-22 dataset there."
        )

    path = csv_files[0]
    print(f"Loading: {path}")

    df = pd.read_csv(path, low_memory=False)

    required = {"Question", "Answer", "Source"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"Original rows: {len(df):,}")
    print("\nOriginal source distribution:")
    print(df["Source"].value_counts().to_string())

    return df


def identify_model(source):
    if pd.isna(source):
        return None

    name = str(source).strip().lower()

    # Match model families without assuming exact
    # model-version names in the CSV.

    if (
        "chatgpt" in name
        or "gpt-" in name
        or name in {"gpt", "openai"}
    ):
        return "GPT"

    if "claude" in name:
        return "Claude"

    if "gemini" in name:
        return "Gemini"

    return None


def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize line endings and excessive whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def prepare_data(df):
    df = df[["Question", "Answer", "Source"]].copy()

    df["model"] = df["Source"].apply(identify_model)

    df = df.dropna(subset=["model"]).copy()

    print("\nMatched family distribution:")
    print(df["model"].value_counts().to_string())

    df = df.rename(
        columns={
            "Question": "input",
            "Answer": "output",
        }
    )

    df["input"] = df["input"].apply(clean_text)
    df["output"] = df["output"].apply(clean_text)

    df = df[
        (df["input"].str.len() > 0)
        & (df["output"].str.len() > 0)
    ].copy()

    # Remove exact repeated prompt-response examples
    # from the same family.
    df = df.drop_duplicates(
        subset=["model", "input", "output"]
    )

    # Avoid identical responses across different
    # model labels, which would introduce ambiguity.
    conflicting_outputs = (
        df.groupby("output")["model"]
        .nunique()
    )
    conflicting_outputs = set(
        conflicting_outputs[conflicting_outputs > 1].index
    )

    df = df[
        ~df["output"].isin(conflicting_outputs)
    ].copy()

    print("\nDistribution after cleaning:")
    print(df["model"].value_counts().to_string())

    missing_families = set(LABELS) - set(df["model"])

    if missing_families:
        raise ValueError(
            f"No samples found for: {missing_families}. "
            "Check the Source values printed above "
            "and update identify_model()."
        )

    # Sample the same number from each family.
    available = df["model"].value_counts()
    sample_count = min(
        SAMPLES_PER_CLASS,
        int(available.min()),
    )

    if sample_count < 10:
        raise ValueError(
            "Not enough examples per model family "
            "to create reliable splits."
        )

    balanced = []

    for family in LABELS:
        subset = df[df["model"] == family]

        sampled = subset.sample(
            n=sample_count,
            random_state=SEED,
        )

        balanced.append(sampled)

    df = pd.concat(
        balanced,
        ignore_index=True,
    )

    df["label"] = df["model"].map(LABELS)

    # Normalize prompts for grouping.
    # The same prompt should stay in one split.
    df["prompt_group"] = (
        df["input"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    print("\nFinal balanced distribution:")
    print(df["model"].value_counts().to_string())

    return df


def split_data(df):
    # Group by prompt to avoid prompt leakage.
    unique_prompts = pd.DataFrame({
        "prompt_group": df["prompt_group"].unique()
    })

    if len(unique_prompts) < 7:
        raise ValueError(
            "Not enough distinct prompts to split."
        )

    train_prompts, temp_prompts = train_test_split(
        unique_prompts["prompt_group"],
        test_size=0.30,
        random_state=SEED,
        shuffle=True,
    )

    val_prompts, test_prompts = train_test_split(
        temp_prompts,
        test_size=0.50,
        random_state=SEED,
        shuffle=True,
    )

    train_set = set(train_prompts)
    val_set = set(val_prompts)
    test_set = set(test_prompts)

    train_df = df[
        df["prompt_group"].isin(train_set)
    ].copy()

    val_df = df[
        df["prompt_group"].isin(val_set)
    ].copy()

    test_df = df[
        df["prompt_group"].isin(test_set)
    ].copy()

    # Ensure no prompt is shared between splits.
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    for name, split in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        if split.empty:
            raise ValueError(
                f"{name} split is empty."
            )

        missing = set(LABELS) - set(split["model"])

        if missing:
            raise ValueError(
                f"{name} split is missing {missing}. "
                "Try increasing SAMPLES_PER_CLASS "
                "or using a different seed."
            )

    return train_df, val_df, test_df


def save_data(train_df, val_df, test_df):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "input",
        "output",
        "model",
        "label",
    ]

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    for name, df in splits.items():
        path = PROCESSED_DIR / f"{name}.csv"

        df[columns].to_csv(
            path,
            index=False,
        )

        print(f"\n{name.upper()}: {len(df):,} rows")
        print(df["model"].value_counts().to_string())
        print(f"Saved to: {path}")

    mapping_path = PROCESSED_DIR / "label_mapping.json"

    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(LABELS, f, indent=4)

    print(f"\nSaved label mapping: {mapping_path}")


def main():
    print("=" * 50)
    print("LLM DETECTOR - DATA PREPARATION")
    print("=" * 50)

    df = load_data()
    df = prepare_data(df)

    train_df, val_df, test_df = split_data(df)

    save_data(train_df, val_df, test_df)

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
