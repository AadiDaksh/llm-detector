"""For RQ4 - What characteristics distinguish different LLMs?"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAMES = ["GPT", "Claude", "Gemini"]

FEATURES = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "vocabulary_diversity",
    "avg_word_length",
    "punctuation_per_100_words",
    "commas_per_100_words",
    "exclamations_per_100_words",
    "questions_per_100_words",
]

"""Extract lowercase words from a response."""
def tokenize_words(text): 
    return re.findall(r"\b\w+\b", str(text).lower())


def extract_features(text):
    """Extract linguistic features from one LLM response."""
    text = str(text)
    words = tokenize_words(text)

    word_count = len(words)
    unique_words = len(set(words))

    # Count sentences using sentence-ending punctuation.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    sentence_count = len(sentences)

    # Calculate average sentence and word lengths.
    avg_sentence_length = (
        word_count / sentence_count
        if sentence_count > 0 else 0
    )

    avg_word_length = (
        sum(len(word) for word in words) / word_count
        if word_count > 0 else 0
    )

    # Calculate the proportion of unique words.
    vocabulary_diversity = (
        unique_words / word_count
        if word_count > 0 else 0
    )

    # Count punctuation and normalize by response length.
    punctuation_count = len(
        re.findall(r"[.,!?;:]", text)
    )

    commas = text.count(",")
    exclamations = text.count("!")
    questions = text.count("?")

    scale = 100 / word_count if word_count > 0 else 0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "vocabulary_diversity": vocabulary_diversity,
        "avg_word_length": avg_word_length,
        "punctuation_per_100_words": punctuation_count * scale,
        "commas_per_100_words": commas * scale,
        "exclamations_per_100_words": exclamations * scale,
        "questions_per_100_words": questions * scale,
    }


def load_data(data_dir, split):
    """Load the selected processed dataset splits."""
    frames = []

    for name in split:
        path = data_dir / f"{name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run prepare_data.py first."
            )

        df = pd.read_csv(path)

        required = {"output", "model"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"{path} is missing columns: "
                f"{required - set(df.columns)}"
            )

        df = df.dropna(subset=["output", "model"]).copy()
        df["split"] = name
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Keep only the three LLM families in this project.
    data = data[data["model"].isin(MODEL_NAMES)].copy()

    if data.empty:
        raise ValueError("No valid GPT, Claude, or Gemini responses found.")

    return data


def analyze_responses(data):
    """Calculate linguistic features for every response."""
    feature_rows = []

    for text in data["output"]:
        feature_rows.append(extract_features(text))

    features = pd.DataFrame(feature_rows)

    result = pd.concat(
        [data.reset_index(drop=True), features],
        axis=1
    )

    return result


def summarize_features(data):
    """Calculate average linguistic features for each LLM."""
    summary = data.groupby("model")[FEATURES].agg(
        ["mean", "std", "median"]
    )

    summary = summary.round(4)

    return summary


def plot_features(data, save_dir):
    """Generate comparison charts for linguistic features."""
    chart_features = {
        "word_count": "Average Response Length",
        "avg_sentence_length": "Average Sentence Length",
        "vocabulary_diversity": "Vocabulary Diversity",
        "avg_word_length": "Average Word Length",
        "punctuation_per_100_words": "Punctuation per 100 Words",
    }

    averages = data.groupby("model")[list(chart_features)].mean()
    averages = averages.reindex(MODEL_NAMES)

    for feature, title in chart_features.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        values = averages[feature]

        ax.bar(
            values.index,
            values.values
        )

        ax.set_title(title)
        ax.set_xlabel("LLM Family")
        ax.set_ylabel(title)

        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        plt.tight_layout()

        figure_path = save_dir / f"{feature}.png"
        plt.savefig(figure_path, dpi=300)
        plt.close(fig)


def analyze_common_words(data, save_dir, top_n=20):
    """Find the most frequent words used by each LLM."""
    common_words = {}

    stopwords = {
        "the", "a", "an", "and", "or", "but", "is", "are",
        "was", "were", "be", "been", "to", "of", "in",
        "on", "for", "with", "as", "at", "by", "from",
        "it", "this", "that", "these", "those", "i",
        "you", "your", "we", "they", "he", "she",
        "not", "can", "will", "would", "could", "should",
        "have", "has", "had", "do", "does", "did"
    }

    for model_name in MODEL_NAMES:
        responses = data.loc[
            data["model"] == model_name,
            "output"
        ]

        counter = Counter()

        for response in responses:
            words = tokenize_words(response)

            counter.update(
                word for word in words
                if word not in stopwords and len(word) > 2
            )

        common_words[model_name] = [
            {"word": word, "count": count}
            for word, count in counter.most_common(top_n)
        ]

    path = save_dir / "common_words.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(common_words, file, indent=4)

    return common_words


def main():
    parser = argparse.ArgumentParser(
        description="Analyze linguistic fingerprints of LLM responses."
    )

    parser.add_argument(
        "--data-dir",
        default="data/processed"
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test", "all"],
        default="train"
    )

    parser.add_argument(
        "--top-words",
        type=int,
        default=20
    )

    args = parser.parse_args()

    if args.top_words < 1:
        parser.error("top-words must be positive")

    data_dir = Path(args.data_dir)

    save_dir = Path("results/analysis") / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select which dataset splits to analyze.
    splits = (
        ["train", "val", "test"]
        if args.split == "all"
        else [args.split]
    )

    # Load the original response text.
    data = load_data(data_dir, splits)

    print(f"Analyzing {len(data)} responses")

    print("\nResponses per LLM:")
    print(data["model"].value_counts())

    # Extract linguistic features.
    analyzed_data = analyze_responses(data)

    features_path = save_dir / "linguistic_features.csv"
    analyzed_data.to_csv(features_path, index=False)

    # Calculate and save summary statistics.
    summary = summarize_features(analyzed_data)

    summary_path = save_dir / "feature_summary.csv"
    summary.to_csv(summary_path)

    print("\nAverage linguistic features:")
    print(
        analyzed_data.groupby("model")[FEATURES]
        .mean()
        .round(3)
    )

    # Generate charts and word-frequency statistics.
    plot_features(analyzed_data, save_dir)

    common_words = analyze_common_words(
        analyzed_data,
        save_dir,
        top_n=args.top_words
    )

    print("\nMost frequently used words:")

    for model_name, words in common_words.items():
        top_words = [item["word"] for item in words[:10]]
        print(f"{model_name}: {', '.join(top_words)}")

    print(f"\nResults saved to: {save_dir}")


if __name__ == "__main__":
    main()
