
# LLM Detector — Who Wrote It?

A project developed as part of **CMPSC 448** at **Pennsylvania State University**.

The project investigates whether neural networks can identify which LLM generated a text response. Using linguistic and structural patterns, the system classifies responses from **GPT, Claude, and Gemini** through Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) architectures.The full project report is available here:
[View Project Report (PDF)](report/CMPSC448_Project1_Report.pdf)

## Features

- Classifies AI-generated text from GPT, Claude, and Gemini
- Implements CNN and Bidirectional LSTM classifiers using PyTorch
- Supports input-only, output-only, and combined text classification
- Processes and tokenizes text into numerical sequences
- Uses prompt-aware training, validation, and test splits
- Supports configurable epochs, batch size, learning rate, and random seed
- Saves trained checkpoints based on validation accuracy
- Evaluates accuracy, precision, recall, and F1-score
- Generates confusion matrices and prediction reports
- Analyzes linguistic fingerprints across LLM families

## Tech Stack

- **Language:** Python
- **Deep Learning:** PyTorch
- **Architectures:** CNN, Bidirectional LSTM
- **Data Processing:** Pandas, NumPy
- **Evaluation:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Development:** Git, GitHub

## Dataset

This project uses the **TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification** dataset.

The original dataset contains 66,000 samples across 22 classes. For this project, a balanced subset of 6,000 examples is selected from three LLM families.

| LLM Family | Samples |
|------------|---------|
| GPT | 2,000 |
| Claude | 2,000 |
| Gemini | 2,000 |
| **Total** | **6,000** |

### Download the Dataset

The raw dataset is not included in this repository due to its size.

1. Visit the official [TXD-22 dataset page on Mendeley Data](https://data.mendeley.com/datasets/prcjcggtjf/1).
2. Click **Download** to obtain the dataset.
3. Extract the downloaded archive if necessary.
4. Locate the CSV file named `TXD-22 A Large-Scale Benchmark Dataset.csv`.
5. Place the CSV inside the `data/raw/` directory.

The directory should look like this:

```text
llm-detector/
└── data/
    └── raw/
        └── TXD-22 A Large-Scale Benchmark Dataset.csv
```

**Dataset source:** Iqbal, M. S., & Kashem, M. A. (2026). *TXD-22: A Large-Scale Benchmark Dataset for Multi-Class AI-Generated and Mixed-Source Text Identification*. Mendeley Data, Version 1.

**DOI:** https://doi.org/10.17632/prcjcggtjf.1

## Setup

### 1. Clone the Repository

```powershell
git clone https://github.com/AadiDaksh/llm-detector.git
cd llm-detector
```

### 2. Create and Activate a Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
py -m pip install -r requirements.txt
```

### 4. Prepare the Dataset

After downloading the dataset and placing it in `data/raw/`, run:

```powershell
py prepare_data.py
```

This selects the relevant LLM families, processes the data, and generates the training, validation, and testing datasets.

Generated files:

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
└── label_mapping.json
```

The processed CSV files are generated locally and are not required to be downloaded separately.

## Model Architectures

### Convolutional Neural Network (CNN)

The CNN classifier identifies local linguistic patterns using convolutional filters over word embeddings.

- Trainable word embeddings
- Parallel convolutional filters with kernel sizes 3, 4, and 5
- Max-pooling operations
- Dropout regularization
- Fully connected classification layer

### Bidirectional LSTM

The Bidirectional LSTM processes text in forward and backward directions to capture contextual and sequential relationships.

- 128-dimensional trainable word embeddings with padding index 0
- Single-layer bidirectional LSTM, hidden size 128 per direction (256 total)
- Masked mean pooling and masked max pooling over the sequence, concatenated into a 512-dimensional representation (padding positions excluded from both pools)
- Dropout regularization (p = 0.3)
- Fully connected classification layer (512 → 3 classes)

## Training

The `train.py` script trains either architecture using one of three classification modes.

### Classification Modes

| Mode | Description |
|------|-------------|
| `input` | Classifies using only the original prompt |
| `output` | Classifies using only the generated response |
| `both` | Classifies using the prompt and response together |

### Basic Training

Train a CNN using generated responses:

```powershell
py train.py --model cnn --mode output --epochs 10
```

Train a Bidirectional LSTM:

```powershell
py train.py --model lstm --mode output --epochs 10
```

### Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Architecture: `cnn` or `lstm` | See `--help` |
| `--mode` | Configuration: `input`, `output`, or `both` | See `--help` |
| `--epochs` | Number of training epochs | 10 |
| `--batch-size` | Training batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--seed` | Random seed | 42 |

### Custom Training Examples

Train a CNN with a custom batch size and learning rate:

```powershell
py train.py --model cnn --mode output --epochs 15 --batch-size 16 --lr 0.0005
```

Train an LSTM using both the prompt and response:

```powershell
py train.py --model lstm --mode both --epochs 10 --batch-size 32 --lr 0.001 --seed 42
```

Train a CNN using only input prompts:

```powershell
py train.py --model cnn --mode input --epochs 5 --batch-size 64
```

Run a short LSTM training experiment:

```powershell
py train.py --model lstm --mode output --epochs 1 --batch-size 32
```

The best checkpoint is saved according to validation accuracy.

Example checkpoint locations:

```text
results/
├── cnn/
│   ├── input_best.pt
│   ├── output_best.pt
│   └── both_best.pt
└── lstm/
    ├── input_best.pt
    ├── output_best.pt
    └── both_best.pt
```

Only checkpoints corresponding to completed training runs will exist. Training an existing configuration may replace its previous best checkpoint.

## Evaluation

The `evaluate.py` script loads a trained checkpoint and evaluates it on the held-out test dataset.

Evaluate the CNN:

```powershell
py evaluate.py --model cnn --mode output
```

Evaluate the LSTM:

```powershell
py evaluate.py --model lstm --mode output
```

Evaluate other configurations:

```powershell
py evaluate.py --model cnn --mode input
py evaluate.py --model cnn --mode both
py evaluate.py --model lstm --mode input
py evaluate.py --model lstm --mode both
```

A compatible checkpoint must exist before running each evaluation.

### Evaluation Outputs

The evaluation script generates:

- Test accuracy
- Precision, recall, and F1-score for each LLM family
- Classification report
- Confusion matrix
- JSON performance metrics
- CSV containing individual predictions

Example:

```text
results/cnn/
├── output_best.pt
├── output_metrics.json
├── output_confusion_matrix.png
└── output_predictions.csv
```

## Linguistic Fingerprint Analysis

The `analyze.py` script examines linguistic differences between responses generated by GPT, Claude, and Gemini.

Run:

```powershell
py analyze.py
```

The analysis investigates:

- Response length
- Vocabulary diversity
- Average sentence length
- Punctuation patterns
- Common word usage

Generated analysis outputs are stored in `results/analysis/`.

## Results

The following results were obtained on a held-out test set of 904 examples.

| Model | Configuration | Test Accuracy | Macro-F1 |
|-------|---------------|---------------|----------|
| CNN  | Input only       | 33.41% | 0.264 |
| CNN  | Output only      | **94.36%** | 0.942 |
| LSTM | Input only       | 33.41% | 0.265 |
| LSTM | Output only      | **92.26%** | 0.922 |
| LSTM | Input + output   | 90.60% | 0.906 |

The CNN output-only model achieved the highest accuracy among the reported experiments, with the LSTM output-only configuration close behind.

Both input-only models performed near the random classification baseline of approximately 33.33%, suggesting that generated responses contain substantially more information about the source LLM than the original prompt. On TXD-22 the user's prompt is sent to every LLM, so the prompt alone carries no evidence about which model answered.

Adding the prompt to the response (`input + output`) slightly decreased LSTM accuracy relative to `output only`. One likely explanation is that uninformative prompt tokens occupy sequence positions the LSTM still has to process, acting as a mild distractor.

Each reported CNN checkpoint was selected after 10 training epochs. Each reported LSTM checkpoint was selected after 8 training epochs based on the best validation accuracy.

## Project Structure

```text
llm-detector/
├── data/
├── models/
├── rq3/
│   ├── rq3_experiment.py
│   └── results/
├── reports/
│   └── CMPSC448_Project1_Report.pdf
├── results/
├── prepare_data.py
├── data_pipeline.py
├── train.py
├── evaluate.py
├── analyze.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- Classification currently focuses on three LLM families.
- Model performance depends on the dataset's characteristics and distribution.
- Results may not generalize to unseen LLM versions or different datasets.
- High benchmark accuracy does not guarantee reliable attribution of arbitrary AI-generated text.

