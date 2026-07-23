# Amazon Electronics Sentiment Analysis & Review Summarization

**Group 10** | Sapna Patel · Prema Parks · Yo Furusawa · Jose Juan Enriquez Valdes

---

## Project Overview

Online shopping platforms contain millions of customer reviews that provide valuable information about product quality, customer satisfaction, common product strengths, and recurring product issues. However, manually reading hundreds or thousands of reviews before making a purchasing decision is time-consuming and impractical.

The goal of this project is to develop a Natural Language Processing system that automatically:

1. Classifies Amazon Electronics reviews as positive, neutral, or negative.
2. Generates concise product-level summaries from collections of customer reviews.
3. Compares alternative summarization strategies to determine whether separating reviews by sentiment improves summary quality.

The project combines two transformer-based language models:

1. **Sentiment Classification** — Fine-tune `bert-base-uncased` to classify reviews as positive, neutral, or negative using star-rating-derived labels.
2. **Review Summarization** — Use `facebook/bart-large-cnn` to generate abstractive summaries of grouped product reviews.

The summarization pipeline supports sentiment-separated review groups and is designed to compare them with an all-reviews-combined strategy.

**Dataset:** [Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), published by the McAuley Lab at the University of California, San Diego.

The full Electronics dataset contains approximately 18.3 million reviews across approximately 1.6 million products. To keep the project computationally manageable, the team uses sampled and processed subsets for development, training, evaluation, and feasibility testing.

---

## Research Questions

### RQ1

How accurately can a fine-tuned BERT model classify Amazon Electronics reviews into positive, neutral, and negative sentiment using star-rating-derived labels?

### RQ2

How does class balancing influence BERT classification performance, particularly recall and F1-score for neutral reviews?

### RQ3

How does separating reviews by sentiment before summarization affect the quality of BART-generated summaries compared with summarizing all reviews together?

---

## Dataset

### Original Dataset

The original Amazon Electronics dataset contains approximately:

- 18.3 million reviews
- 1.6 million products
- Product identifiers
- Product metadata
- Review titles
- Review bodies
- Star ratings
- Helpful-vote information
- Timestamps and additional review metadata

### Project Dataset

To reduce computational requirements, the data pipeline creates a manageable project sample from the original Electronics dataset.

The pipeline:

- Selects a subset of products from the metadata dataset.
- Filters products based on the minimum number of available reviews.
- Samples a manageable number of review records.
- Merges review data with product metadata.
- Cleans and validates the review content.
- Creates sentiment labels from star ratings.
- Produces training, validation, and test datasets.

The exact sample size and filtering parameters should be defined in the project configuration and may be adjusted during experimentation.

### Review Features

The processed dataset includes fields such as:

- Product ID
- Product title
- Review title
- Review text
- Combined input text
- Star rating
- Sentiment label
- Helpful votes
- Product metadata

During preprocessing, review titles and review bodies are combined into a single text field for sentiment classification and summarization.

---

## Sentiment Labels

Reviews are assigned sentiment labels based on their original star ratings.

| Star Rating | Sentiment Label | Numeric Label |
|-------------|-----------------|--------------:|
| 1–2 stars   | Negative        |       0       |
| 3 stars     | Neutral         |       1       |
| 4–5 stars   | Positive        |       2       |

These labels are used for:

- BERT training and evaluation
- Sentiment-separated review grouping
- BART summary generation
- Sentiment-alignment evaluation

---

## Repository Structure

```text
amazon-electronics-sentiment-analysis/
├── configs/
│   ├── data_config.yaml          # Sampling, split, and preprocessing settings
│   └── model_config.yaml         # BERT hyperparameters + BART generation params ("bert"/"bart" sections)
│
├── data/
│   ├── raw/
│   │   ├── metadata_sample.csv
│   │   └── reviews_sample.csv
│   │
│   └── processed/
│       ├── preprocessed_reviews.csv
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
│
├── docs/                         # Literature review and model documentation
│
├── experiments/
│   └── bert_class_balancing.ipynb # Reproducible baseline-vs-weighted BERT run (RQ2)
│
├── models/                       # Saved model checkpoints (git-ignored)
│
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
│
├── outputs/
│   ├── bert_evaluation_baseline.{txt,json}   # BERT baseline results
│   ├── bert_evaluation_weighted.{txt,json}   # BERT class-weighted results
│   ├── bert_sample_predictions_baseline.csv  # 10 sample BERT predictions (baseline)
│   ├── bert_sample_predictions_weighted.csv  # 10 sample BERT predictions (weighted)
│   ├── summary_samples.csv
│   ├── summary_evaluation.csv
│   ├── strategy_comparison.csv
│   └── description.md            # What's in this directory and its status
│
├── src/
│   ├── data_loader.py            # Sample raw review and metadata data
│   ├── preprocess.py             # Clean, label, and split sampled data
│   ├── train.py                  # BERT fine-tuning script
│   ├── evaluate.py               # BERT metrics (accuracy, F1, confusion matrix) + BART summary evaluation
│   ├── model_runner.py           # BART summarization pipeline
│   ├── summarizer.py             # BART summary generation
│   └── utils/
│       ├── __init__.py           # Configuration loading
│       └── helpers.py            # Shared text cleaning and label mapping
│
├── requirements.txt
└── README.md
```

### Important Repository Notes

- The `models/` directory may be excluded from GitHub because trained model checkpoints can be large.
- The `outputs/` directory contains generated summaries and evaluation results.
- The current BART pipeline reads the processed dataset from:

```text
data/processed/preprocessed_reviews.csv
```

- If the processed dataset is moved or renamed, the path in `configs/model_config.yaml`'s `bart.data.processed_path` must also be updated.

---

## Technology Stack

The project uses the following frameworks and libraries:

- Python
- pandas
- NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers
- BERT
- BART
- NLTK
- VADER Sentiment
- ROUGE Score
- Matplotlib
- Jupyter Notebook
- YAML configuration files
- Git and GitHub

---

## Prerequisites

Before running the project, ensure that the following software is installed:

- Python 3.10 or later
- Git
- Visual Studio Code, Jupyter Notebook, or another Python development environment
- Internet access for the initial model download
- Sufficient local storage for datasets, Python packages, and model files

A Python virtual environment is strongly recommended to isolate project dependencies from the system Python installation.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd amazon-electronics-sentiment-analysis
```

Replace `<repo-url>` with the GitHub repository URL.

---

### 2. Create a Virtual Environment

The project uses `.venv` as the recommended virtual-environment folder.

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

When the environment is active, the terminal prompt should begin with:

```text
(.venv)
```

For example:

```text
(.venv) PS C:\Projects\amazon-electronics-sentiment-analysis>
```

---

### Windows PowerShell Troubleshooting

Some Windows systems block PowerShell scripts by default.

If the following activation command fails:

```powershell
.\.venv\Scripts\Activate.ps1
```

and PowerShell displays an execution-policy error, temporarily allow script execution for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

This bypass applies only to the current PowerShell session. It does not permanently change the system-wide execution policy.

The virtual environment can also be used without activation by calling its Python executable directly:

```powershell
.\.venv\Scripts\python.exe src\model_runner.py
```

---

### 3. Install Dependencies

After activating the virtual environment, upgrade `pip` and install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Verify the Environment

Run:

```bash
python -c "import pandas, numpy, sklearn, torch, transformers; print('Environment ready')"
```

Expected output:

```text
Environment ready
```

If a package is missing, confirm that the virtual environment is active and reinstall the requirements:

```bash
pip install -r requirements.txt
```

---

### 5. Download the VADER Lexicon

The summary-evaluation module uses the VADER sentiment lexicon.

Run:

```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

---

## Configuration

Project settings are stored in the `configs/` directory.

### `data_config.yaml`

Contains configuration for the data sampling and preprocessing pipeline (`data_loader.py`/`preprocess.py`), including:

- metadata and review sample sizes
- minimum reviews required per product
- train/validation split ratios
- minimum review length (short reviews are dropped)
- random seed

### `model_config.yaml`

Both models' hyperparameters and paths live in this single file, under top-level `bert:` and `bart:` sections.

**`bert:`** contains settings for sentiment classification, including:

- Pretrained model name
- Maximum sequence length
- Learning rate
- Batch size
- Number of epochs
- Warmup steps
- Weight decay
- Random seed
- Dataset paths
- Model-checkpoint directory
- Evaluation-output directory
- Class-balancing parameters

**`bart:`** contains settings for review summarization, including:

- Pretrained summarization model
- Processed dataset path
- Product-column name
- Product-title column
- Review-title column
- Review-text column
- Rating column
- Sentiment column
- Helpful-vote column
- Maximum input-token length
- Minimum summary length
- Maximum summary length
- Beam-search parameters
- Length penalty
- Repetition controls
- Number of reviews selected per group
- Number of representative products
- Summary-output path
- Evaluation-output path
- Strategy-comparison path

The current processed dataset path (used by both `bert.data` and `bart.data`) is:

```text
data/processed/preprocessed_reviews.csv
```

If the dataset is moved or renamed, update the following configuration value under `bart:`:

```yaml
bart:
  data:
    processed_path: data/processed/preprocessed_reviews.csv
```

---

## Usage

The project follows a sequential workflow from data preparation to sentiment classification and review summarization.

Run all commands from the repository root.

---

## Step 1: Load and Sample the Dataset

Run:

```bash
python src/data_loader.py
```

The data-loading process is designed to:

- Load the Amazon Reviews 2023 Electronics review dataset.
- Load the associated product metadata.
- Validate required fields.
- Report the total number of rows.
- Report missing-value counts.
- Report the star-rating distribution.
- Select a manageable subset of products.
- Filter products that do not contain enough reviews.
- Save sampled review and metadata files.

Expected sampled outputs may include:

```text
data/raw/reviews_sample.csv
data/raw/metadata_sample.csv
```

---

## Step 2: Preprocess the Dataset

Run:

```bash
python src/preprocess.py
```

The preprocessing script is designed to:

- Load the sampled review and metadata files.
- Merge reviews with product metadata.
- Remove duplicate records.
- Remove incomplete records.
- Remove missing review text.
- Remove extremely short reviews.
- Clean and normalize review text.
- Combine review titles and review bodies.
- Convert star ratings into sentiment labels.
- Filter products that do not contain enough reviews.
- Split the processed dataset into training, validation, and test sets.
- Save the processed files under `data/processed/`.

The generated files include:

```text
data/processed/
├── preprocessed_reviews.csv
├── train.csv
├── validation.csv
└── test.csv
```

These files are used for exploratory data analysis, model training, evaluation, and summarization.

---

## Step 3: Exploratory Data Analysis

Run:

```bash
jupyter notebook notebooks/eda.ipynb
```

The exploratory data analysis examines:

- Star-rating distribution
- Sentiment-class distribution
- Positive, neutral, and negative review balance
- Review-text length
- Average review length
- Reviews per product
- Products with sufficient reviews
- Missing-value patterns
- Duplicate-record patterns
- Additional analyses supporting the research questions

The EDA results help determine whether the processed dataset is suitable for BERT training and product-level summarization.

---

## Step 4: Train the BERT Sentiment Classifier

The BERT training pipeline is intended to fine-tune:

```text
bert-base-uncased
```

for three-class sentiment classification.

Run:

```bash
python src/train.py
```

The training script:

- Loads the processed training dataset.
- Loads the processed validation dataset.
- Tokenizes review text.
- Fine-tunes `bert-base-uncased`.
- Predicts negative, neutral, or positive sentiment.
- Applies configurable training hyperparameters.
- Supports a class-balancing experiment (RQ2): setting `bert.training.use_class_weights: true` in `configs/model_config.yaml` weights the loss function by inverse class frequency (via scikit-learn's `compute_class_weight(class_weight="balanced", ...)`, computed from the actual train-split label distribution), so mistakes on the minority neutral class are penalized more heavily during training.
- Saves the trained model checkpoint and tokenizer.

Model checkpoints are saved under:

```text
models/bert_sentiment_baseline/    # training.use_class_weights: false
models/bert_sentiment_weighted/    # training.use_class_weights: true
```

### Reproducing the BERT Training Run

`experiments/bert_class_balancing.ipynb` runs both variants end to end (baseline, then class-weighted) and produces the comparison used to answer RQ2. It's designed to work on any CUDA GPU without assuming anything is pre-installed in the notebook's own kernel environment — it creates an isolated Python virtual environment under `~/.amazon_sentiment_venv` and installs all project dependencies there, which avoids conflicts with shared or read-only cluster environments. Open the notebook and run all cells top to bottom; each variant's `python src/train.py` + `python src/evaluate.py` run takes roughly 8–9 minutes on an NVIDIA V100.

---

## Step 5: Evaluate the BERT Sentiment Classifier

Run:

```bash
python src/evaluate.py
```

The BERT evaluation process calculates, on the held-out test split (7,239 reviews: 1,390 negative, 483 neutral, 5,366 positive):

- Accuracy
- Precision (macro and per-class)
- Recall (macro and per-class)
- F1-score (macro, weighted, and per-class)
- Confusion matrix

Results are written to `outputs/bert_evaluation_{baseline,weighted}.txt` (a full `sklearn` classification report) and the equivalent structured `.json`, and are also committed to this repository — see [Preliminary BERT Results](#preliminary-bert-results) below.

A random 10-row sample of test-set predictions (review text, actual sentiment, predicted sentiment) is also written to `outputs/bert_sample_predictions_{baseline,weighted}.csv`, for manual spot-checking of what the model gets right and wrong.

---

## Preliminary BERT Results

Both variants were trained for 3 epochs on an NVIDIA V100 GPU and evaluated on the same 7,239-review test split.

| Metric | Baseline | Class-Weighted | Delta |
|---|---:|---:|---:|
| Accuracy | 0.9058 | 0.9039 | -0.0019 |
| Precision (macro) | 0.7543 | 0.7486 | -0.0057 |
| Recall (macro) | 0.7511 | 0.7692 | +0.0180 |
| F1 (macro) | 0.7527 | 0.7580 | +0.0053 |
| F1 (weighted) | 0.9056 | 0.9066 | +0.0010 |
| F1 (negative) | 0.8470 | 0.8528 | +0.0058 |
| F1 (neutral) | 0.4493 | 0.4607 | **+0.0114** |
| F1 (positive) | 0.9619 | 0.9607 | -0.0012 |
| Recall (neutral) | 0.4493 | 0.4969 | **+0.0476** |

**RQ1 answer:** the baseline model reaches 90.6% overall accuracy, but performance is very uneven across classes — F1 is 0.96 for positive and 0.85 for negative, but only 0.45 for neutral. Overall accuracy alone substantially overstates how well the model handles the minority neutral class, since positive reviews dominate the test set (5,366 of 7,239 rows).

**RQ2 answer:** class weighting improves neutral-class performance (F1 +0.0114, recall +0.0476) and macro-F1 (+0.0053), but this run shows that gain coming at a small cost to overall accuracy (-0.0019) and macro precision (-0.0057), plus a drop in positive-class recall (-0.0116) and neutral-class precision (-0.0199). Class balancing still helps the specific problem it targets (neutral recall/F1), but neutral reviews remain the hardest class to classify by a wide margin even after weighting, and the size — and even the sign — of the accuracy/precision trade-off varies between runs of the same seed (see [Known Limitations](#known-limitations) and [Reproducibility Notes](#reproducibility-notes) below for why).

Confusion matrices (rows = true label, columns = predicted label):

**Baseline**

| | negative | neutral | positive |
|---|---:|---:|---:|
| **negative** | 1168 | 144 | 78 |
| **neutral** | 128 | 217 | 138 |
| **positive** | 72 | 122 | 5172 |

**Class-Weighted**

| | negative | neutral | positive |
|---|---:|---:|---:|
| **negative** | 1193 | 139 | 58 |
| **neutral** | 139 | 240 | 104 |
| **positive** | 76 | 180 | 5110 |

The weighted model correctly classifies 23 more neutral reviews (240 vs. 217) than the baseline, at the cost of a few more negative/positive misclassifications elsewhere and a small (-0.0019) drop in overall accuracy — consistent with the metrics above.

---

## Step 6: Generate Product Review Summaries

Run the complete BART summarization pipeline:

```bash
python src/model_runner.py
```

The script performs the following steps automatically:

1. Loads the processed review dataset specified in `configs/model_config.yaml`'s `bart` section.
2. Validates the required product, review, rating, and sentiment columns.
3. Selects representative products.
4. Groups reviews by product.
5. Separates review groups by sentiment category.
6. Prioritizes reviews using helpful-vote information when available.
7. Formats selected reviews into model-ready input.
8. Loads `facebook/bart-large-cnn`.
9. Generates combined-review, positive, neutral, and negative summaries.
10. Saves generated summaries under `outputs/`.
11. Runs the summary-evaluation module.
12. Saves detailed and aggregated evaluation results.

The pipeline requires no manual intervention after execution begins.

### Example Execution

```text
Loaded 48,252 reviews.
Selected 20 products.
Prepared 80 summary input documents.

Loading facebook/bart-large-cnn...
BART model loaded.

Generating 80 summaries...

Summaries saved to: outputs/summary_samples.csv
Evaluation saved to: outputs/summary_evaluation.csv
Strategy comparison saved to: outputs/strategy_comparison.csv

Pipeline completed.
Successful summaries: 80
Failed summaries: 0
```

#### Current BART Test Scope

The latest expanded evaluation processed:

- **48,252 processed reviews**
- **20 representative products**
- **80 generated summaries**
- **80 successful summaries**
- **0 failed summaries**

For each product, the pipeline generated:

- 1 combined-review summary
- 1 positive-review summary
- 1 neutral-review summary
- 1 negative-review summary

This produced:

- **20 combined-review summaries**
- **60 sentiment-separated summaries**

The expanded evaluation provides a broader assessment of the summarization
pipeline across a more diverse set of products, increasing confidence in the
consistency and robustness of the generated summaries.

---

## Step 7: Evaluate Generated Summaries

BART summary evaluation runs automatically when the following command is executed:

```bash
python src/model_runner.py
```

The evaluation logic is implemented in:

```text
src/evaluate.py
```

The module calculates:

- Source word count
- Generated-summary word count
- Compression ratio
- Lexical coverage
- Novelty ratio
- Repetition ratio
- Sentiment alignment
- ROUGE-1 source coverage
- ROUGE-2 source coverage
- ROUGE-L source coverage

The evaluation results are saved to:

```text
outputs/summary_evaluation.csv
outputs/strategy_comparison.csv
```

The evaluation output also includes fields for manual qualitative scoring:

- Relevance
- Coherence
- Conciseness
- Pros-and-cons coverage

These fields can be evaluated using a consistent scale such as:

| Score | Interpretation |
|------:|----------------|
|   1   |      Poor      |
|   2   |      Weak      |
|   3   |   Acceptable   |
|   4   |      Good      |
|   5   |    Excellent   |

Because the Amazon review dataset does not contain human-written reference summaries, ROUGE is currently used as a source-coverage diagnostic rather than as a traditional reference-summary score.

---

## Generated Outputs

The BART pipeline creates the following files.

### `outputs/summary_samples.csv`

Contains:

- Product identifier
- Product title
- Summarization strategy
- Sentiment group
- Source-review text
- Generated summary
- Generation status
- Error information, when applicable

### `outputs/summary_evaluation.csv`

Contains summary-level evaluation values, including:

- Summary length
- Source length
- Compression ratio
- Lexical coverage
- Novelty
- Repetition
- Sentiment alignment
- ROUGE source-coverage values
- Manual-evaluation fields

### `outputs/strategy_comparison.csv`

Contains aggregated metrics used to compare summarization strategies.

---

## Preliminary BART Results

The latest expanded evaluation assessed **20 representative products** using both
combined-review and sentiment-separated summarization strategies. A total of
**80 summaries** were generated, including **20 combined-review summaries**
and **60 sentiment-separated summaries**, with no generation failures.

|            Metric          | Combined | Sentiment-Separated |
|----------------------------|---------:|--------------------:|
|     Summaries generated    |    20    |          60        |
|     Successful summaries   |    20    |          60         |
|      Failed summaries      |     0    |           0         |
|  Average source word count |   672.65 |        562.15       |
| Average summary word count |   42.15  |         40.02       |
|      Compression ratio     |   0.0625 |        0.0907       |
|       Lexical coverage     |   97.59% |        96.35%       |
|        Novelty ratio       |    2.41% |         3.65%       |
|      Repetition ratio      |    7.91% |         9.30%       |
|     Sentiment alignment    |    N/A   |        67.50%       |
|  ROUGE-1 source coverage   |   0.0620 |        0.0898       |
|  ROUGE-2 source coverage   |   0.0551 |        0.0762       |
|  ROUGE-L source coverage   |   0.0580 |        0.0855       |

The pipeline successfully generated all **80 summaries** without any failed
model calls.

The combined-review strategy produced broader product-level overviews by
integrating customer opinions from all sentiment categories. In contrast, the
sentiment-separated strategy generated more focused summaries that clearly
distinguished positive, neutral, and negative customer experiences.

**Note on the combined-strategy numbers:** these reflect a fix to a
truncation bug where the combined-strategy input (built by concatenating
negative, then neutral, then positive review sections) could silently drop
the positive section entirely for any product whose combined text exceeded
BART's ~1024-token input limit -- verified against 18 of the 20 products in
an earlier run. The fix caps each sentiment section to a bounded word budget
before concatenation, which is why the average source word count for
combined summaries dropped substantially (was 1682.3 words) and ROUGE
source-coverage roughly doubled -- the input is now shorter, balanced across
all three sentiments, and the summary covers a larger share of it. The
sentiment-separated strategy was never affected by this bug and its numbers
are essentially unchanged.

Overall, the expanded evaluation showed that sentiment-separated summaries
achieved higher source coverage (ROUGE), greater novelty, and better separation
of product strengths and weaknesses, while combined summaries provided concise
high-level overviews of overall customer opinion.

These findings support **Research Question 3**, suggesting that grouping
reviews by sentiment before summarization improves the clarity and
interpretability of generated product summaries.
---

## Summarization Strategies

The project is designed to compare two summarization strategies.

### Combined Reviews

Reviews from all sentiment categories are grouped into one BART input for each product.

This strategy may provide:

- One overall product summary
- A compact overview of general customer opinion
- Strong representation of frequently repeated product themes

However, positive reviews may dominate products with highly imbalanced ratings.

### Sentiment-Separated Reviews

Reviews are separated into positive, neutral, and negative groups before summarization.

This strategy may provide:

- Clearer strengths
- Clearer weaknesses
- Better representation of neutral feedback
- Easier comparison of different customer perspectives

The latest expanded evaluation compared both the combined-review and
sentiment-separated strategies across **20 representative products**.

The resulting evaluation metrics are stored in
`outputs/strategy_comparison.csv` and are used to compare both approaches in
support of Research Question 3.

---

## Known Limitations

The current implementation has several limitations.

- The neutral class remains difficult to classify even after class weighting: F1 only reaches 0.45–0.46 (vs. 0.85–0.96 for negative/positive) — see [Preliminary BERT Results](#preliminary-bert-results). Class balancing helps, but does not close this gap.
- BERT training/evaluation results reflect a single random seed and a single 70/15/15 train/val/test split; no cross-validation or multiple-seed variance estimate has been run yet.
- `src/train.py` sets `training.seed` but does not force full determinism (`TrainingArguments(full_determinism=True)`), and `fp16` mixed precision is auto-enabled on any CUDA GPU. As a result, two runs with the same `seed: 42` on a GPU are not bit-for-bit reproducible — re-running `experiments/bert_class_balancing.ipynb` will land close to, but not exactly on, the numbers in [Preliminary BERT Results](#preliminary-bert-results) (observed drift so far: accuracy within ~0.2 percentage points, neutral-class F1/recall deltas varying by roughly 2x run to run). The qualitative conclusion (class weighting helps the neutral class at some cost elsewhere) has held across runs; the exact decimal values and the size of the accuracy/precision trade-off have not.
- `facebook/bart-large-cnn` supports a maximum input length of approximately 1,024 tokens.
- Large product-review groups must therefore be filtered, prioritized, or truncated.
- Helpful-vote prioritization may exclude less popular but still relevant observations.
- The current grouping strategy uses sentiment labels derived from star ratings.
- The current BART grouping strategy uses star-rating-derived sentiment labels rather than predictions from the completed fine-tuned BERT classifier.
- Integration with BERT-predicted labels remains a potential future extension.
- BART inference can be slow on a CPU.
- The expanded feasibility evaluation uses 20 representative products. Although this provides broader coverage than the initial pilot, the evaluation remains limited relative to the full   Amazon Electronics dataset.
- Additional products should be tested before drawing broad conclusions.
- The Amazon dataset does not provide human-written reference summaries.
- ROUGE is therefore used as a source-coverage diagnostic.
- Manual qualitative evaluation was performed on representative summaries; however, evaluation remains subjective because no human-written reference summaries are available.
- Generated summaries may occasionally simplify, omit, or overemphasize source-review information.
- The current sentiment-alignment score indicates that not every summary fully preserves its intended sentiment category.

---

## Reproducibility Notes

### BERT Baseline/Weighted Results

To reproduce the results in [Preliminary BERT Results](#preliminary-bert-results):

1. Clone or pull the latest repository (a CUDA GPU is required — training runs in a few minutes on a V100, but is impractically slow on CPU).
2. Open `experiments/bert_class_balancing.ipynb` and run all cells top to bottom. It creates its own isolated virtual environment and installs dependencies there, so no manual environment setup is required, and it works the same way whether you're on a personal GPU machine, a shared/managed cluster, Colab, or Kaggle.
3. The notebook runs `python src/train.py` + `python src/evaluate.py` once with `training.use_class_weights: false` (baseline) and once with `true` (class-weighted), then prints the comparison table shown above.
4. Results are written to `outputs/bert_evaluation_{baseline,weighted}.{txt,json}` and `outputs/bert_sample_predictions_{baseline,weighted}.csv`, and model checkpoints to `models/bert_sentiment_{baseline,weighted}/` (not committed to GitHub — checkpoints are large).
5. **Note:** re-running this on a GPU will not reproduce the exact numbers above bit-for-bit — see the non-determinism bullet in [Known Limitations](#known-limitations). Expect small variation in the decimal values, not an exact match.

### BART Outputs

To reproduce the current BART outputs:

1. Clone or pull the latest repository.
2. Create and activate the `.venv` virtual environment.
3. Install `requirements.txt`.
4. Confirm that the processed dataset exists at:

```text
data/processed/preprocessed_reviews.csv
```

5. Confirm the same path is listed under `bart.data.processed_path` in:

```text
configs/model_config.yaml
```

6. Run:

```bash
python src/model_runner.py
```

7. Review the generated output files under:

```text
outputs/
```

Model files may be downloaded from Hugging Face during the first execution. Internet access may therefore be required the first time the BART pipeline runs.

---

## GitHub Workflow

Before starting work, update the local repository:

```bash
git status
git pull origin main
```

After making changes:

```bash
git add .
git commit -m "Describe the completed changes"
git push origin main
```

If the push is rejected because the remote repository contains newer work:

```bash
git pull --rebase origin main
git push origin main
```

If conflicts appear, resolve the affected files before continuing the rebase.

---

## Troubleshooting

### `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'pandas'
```

This usually means the virtual environment is not active.

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then verify that the prompt begins with:

```text
(.venv)
```

---

### PowerShell Blocks Virtual-Environment Activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### Processed Dataset Not Found

Example:

```text
FileNotFoundError: data/processed_reviews.csv.gz
```

Confirm that the current file exists:

```text
data/processed/preprocessed_reviews.csv
```

Then verify that `configs/model_config.yaml` contains:

```yaml
bart:
  data:
    processed_path: data/processed/preprocessed_reviews.csv
```

---

### BART Model Download Warning

The first execution may display a Hugging Face authentication or rate-limit warning.

The model can usually still be downloaded without authentication. A Hugging Face access token can optionally be configured to improve download reliability and rate limits.

---

### Slow BART Execution

The script may run slowly when using:

```text
Using device: cpu
```

This is expected because `facebook/bart-large-cnn` is a large transformer model.

Reducing the number of products or reviews per group in `configs/model_config.yaml`'s `bart` section can reduce runtime during testing.

---

## Documentation

The `docs/` directory contains supporting project documentation, including:

- Project proposal
- Research questions
- Literature review
- Model-selection justification
- Framework-selection justification
- Methodology and data flow
- Model-pipeline documentation
- Evaluation plans
- Milestone documentation

---

## Team Responsibilities

|          Member           |                                 Primary Responsibility                                          |
|---------------------------|-------------------------------------------------------------------------------------------------|
|        Sapna Patel        |                       Dataset collection and preprocessing pipeline                             |
|        Prema Parks        | Exploratory data analysis, literature review, class-balancing experiments, and GitHub structure |
|        Yo Furusawa        |               BERT sentiment classification, tokenization, fine-tuning, and evaluation          |
| Jose Juan Enriquez Valdes |                     BART summarization and end-to-end model pipeline                            | 

### BART Contribution

The BART summarization contribution includes:

- `configs/model_config.yaml`'s `bart` section
- `src/summarizer.py`
- `src/model_runner.py`
- BART evaluation functions in `src/evaluate.py`
- Product-level review grouping
- Sentiment-separated review grouping
- BART model loading
- Summary generation 
- CSV output generation
- Automatic summary diagnostics
- Strategy evaluation and comparison
- Automatic strategy comparison

---
## Manual Qualitative Evaluation

To complement the quantitative evaluation metrics, a manual assessment was
performed on a representative subset of the generated summaries.

Each summary was evaluated using four qualitative criteria:

- Relevance – Does the summary capture the main ideas from the source reviews?
- Coherence – Is the summary grammatically correct and easy to read?
- Conciseness – Does the summary avoid unnecessary details while preserving key information?
- Pros/Cons Coverage – Does the summary represent the major strengths and weaknesses mentioned by reviewers?

Scores were assigned on a five-point scale (1 = Poor, 5 = Excellent).

|        Strategy      | Relevance | Coherence | Conciseness | Pros/Cons Coverage |
|----------------------|:---------:|:---------:|:-----------:|:------------------:|
|   Combined Reviews   |    4.3    |    4.4    |     4.5     |        4.2         |
|  Sentiment-Separated |    4.8    |    4.7    |     4.8     |        4.7         |

### Observations

- The combined-review strategy generates concise overall product summaries that capture the dominant customer opinions. Some fine-grained details may be omitted due to the large amount of source text.

- The sentiment-separated strategy produces more focused summaries because each generation only processes reviews sharing the same sentiment. This approach provides clearer descriptions of product strengths and weaknesses while maintaining good readability.

Overall, both summarization strategies produced coherent and informative summaries. The sentiment-separated strategy generally performed better in preserving the intended sentiment and presenting more focused product feedback.

## Milestones

|       Milestone      | Due Date |                    Status                         |
|----------------------|----------|---------------------------------------------------|
| M2: Project Proposal |   Jul 5  | Complete                                          |
| M3: Data Pipeline    |   Jul 19 | Complete                                          |
| M4: Model Pipeline   |   Jul 26 | Complete                                          |
| M5: Final Submission |   Aug 9  | Upcoming                                          |

---

## Current Project Status

### Completed or Operational

- GitHub repository established
- Virtual environment configured
- Processed dataset available
- Training, validation, and test datasets created
- Exploratory data-analysis notebook available
- BERT fine-tuning (baseline and class-weighted variants both trained on GPU)
- BERT evaluation (precision, recall, F1, confusion matrix for both variants)
- Class-balancing experiment (RQ2) — see [Preliminary BERT Results](#preliminary-bert-results)
- Reproducible, GPU-environment-agnostic BERT training notebook (`experiments/bert_class_balancing.ipynb`)
- BART configuration created
- BART model successfully loaded
- Product reviews grouped by sentiment
- End-to-end BART pipeline completed
- Expanded BART evaluation completed across 20 representative products
- Eighty BART summaries generated successfully
- Twenty combined-review summaries generated
- Sixty sentiment-separated summaries generated
- Zero BART generation failures
- Combined and sentiment-separated strategy comparison completed
- Summary outputs generated automatically
- Automatic BART evaluation implemented
- ROUGE source-coverage diagnostics completed
- Sentiment-alignment evaluation completed
- Manual qualitative evaluation completed
- Quantitative evaluation using ROUGE and lexical metrics completed
- Research Question 3 evaluation completed
- README documentation completed
- GitHub repository fully synchronized

### In Progress

- Final documentation and presentation materials

---

## License and Academic Use

This repository was developed for academic purposes as part of the Northeastern University Data Analytics Engineering program.

The Amazon Reviews 2023 dataset remains subject to the terms and conditions established by its original publishers.

Pretrained transformer models remain subject to their respective Hugging Face and model-provider licenses.
