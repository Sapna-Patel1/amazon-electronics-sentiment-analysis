# Amazon Electronics Sentiment Analysis & Review Summarization

**Group 10** | Sapna Patel · Prema Parks · Yo Furusawa · Jose Juan Enriquez Valdes

---

## Project Overview

Online shopping platforms contain millions of customer reviews that provide valuable information about product quality, customer satisfaction, and common product issues. However, manually reading hundreds or thousands of reviews before making a purchasing decision is time-consuming and impractical.

The goal of this project is to develop a Natural Language Processing (NLP) system that automatically classifies customer review sentiment and generates concise product summaries from large collections of Amazon Electronics reviews.

The project combines two transformer-based language models:

1. **Sentiment Classification** — Fine-tune `bert-base-uncased` to classify reviews as positive, neutral, or negative using star-rating-derived labels.
2. **Review Summarization** — Use `facebook/bart-large-cnn` to generate abstractive summaries of grouped product reviews, comparing sentiment-separated vs. all-reviews-combined input strategies.

**Dataset:** [Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab, UC San Diego). Full dataset: ~18.3M reviews across ~1.6M products. Working subset: ~276,750 reviews across ~10,000 products.

---

This project investigates the following research questions:

RQ1: How accurately can a fine-tuned BERT model classify Amazon Electronics reviews into positive, neutral, and negative sentiment using star-rating-derived labels?

RQ2: How does class balancing influence BERT classification performance, particularly recall and F1-score for neutral reviews?

RQ3: How does separating reviews by sentiment before summarization affect the quality of BART-generated summaries compared with summarizing all reviews together?

---

## Dataset
The original Electronics dataset contains approximately:

- **18.3 million reviews**
- **1.6 million products**

### Project Dataset

To keep the project computationally manageable, the project uses a sampled subset of approximately **50,000 reviews**. The sampling pipeline:

- randomly selects approximately **10,000 products** from the metadata dataset
- filters out products with fewer than **10 reviews**
- randomly samples approximately **50,000 reviews** from the remaining products

This sampled dataset is used throughout preprocessing, exploratory data analysis, model training, and evaluation.

### Review Features

Each review contains information including:

- Product ID
- Product title
- Review title
- Review text
- Star rating
- Helpful votes
- Product metadata

During preprocessing, **review titles and review text are combined into a single input field** for sentiment classification and summarization.

---

## Sentiment Labels

During preprocessing, reviews are assigned sentiment labels based on their original star ratings. These labels are used for training and evaluating the BERT sentiment classifier.

| Star Rating | Label    | Numeric |
|-------------|----------|---------|
| 1–2 stars   | negative | 0       |
| 3 stars     | neutral  | 1       |
| 4–5 stars   | positive | 2       |

---

## Repository Structure

```
amazon-electronics-sentiment-analysis/
├── data/
│   ├── raw/                      # Sampled raw review/metadata CSVs
│   └── processed/                # Cleaned dataset plus train/validation/test splits
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── experiments/                  # Experimental notebooks and results
├── models/                       # Saved model checkpoints (git-ignored)
├── src/
│   ├── data_loader.py            # Sample raw review/metadata data
│   ├── preprocess.py             # Clean, label, and split the sampled data
│   ├── train.py                  # BERT fine-tuning script
│   ├── evaluate.py               # Evaluation: accuracy, F1, confusion matrix
│   ├── model_runner.py           # BART summarization pipeline
│   ├── summarizer.py             # BART summary generation
│   ├── summary_evaluation.py     # BART summary evaluation
│   └── utils/
│       ├── __init__.py           # load_config()
│       └── helpers.py            # Shared text cleaning, label mapping, data splitting
├── configs/
│   ├── data_config.yaml          # Sampling, split, and preprocessing settings
│   ├── bert_config.yaml          # BERT hyperparameters and paths
│   └── bart_config.yaml          # BART generation parameters and paths
├── outputs/                      # Generated summaries and evaluation results
├── docs/                         # Literature review and model documentation
├── requirements.txt
└── README.md
```

---

## Prerequisites

Before running the project, ensure you have:

- Python 3.10 or later
- Git
- Required Python packages listed in `requirements.txt`

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd amazon-electronics-sentiment-analysis
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download NLTK data (for VADER baseline)

```python
import nltk
nltk.download('vader_lexicon')
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

### `bert_config.yaml`

Contains configuration for BERT sentiment classification, including:

- pretrained model
- maximum sequence length
- learning rate
- batch size
- number of training epochs
- warmup steps
- weight decay
- random seed
- dataset path
- sentiment label mapping
- model and output directories

### `bart_config.yaml`

Contains configuration for BART review summarization, including:

- pretrained summarization model
- maximum input length
- minimum and maximum summary length
- beam search parameters
- review grouping settings
- dataset path
- input column names
- output file locations

---

## Usage

The project follows a sequential pipeline from data sampling to model evaluation.

### Step 1: Sample the Dataset

Run:

```bash
python src/data_loader.py
```

This script:

- loads the raw Amazon Reviews 2023 Electronics review and metadata datasets
- randomly samples approximately **10,000 products**
- filters out products with fewer than **10 reviews**
- randomly samples approximately **50,000 reviews**
- saves the sampled review and metadata datasets as CSV files

---

### Step 2: Preprocess the Dataset

Run:

```bash
python src/preprocess.py
```

This script:

- merges the sampled review and metadata datasets
- removes duplicate and incomplete records
- cleans and formats review text
- combines review titles and review text into a single input field
- assigns sentiment labels based on star ratings
- splits the processed dataset into **training (70%)**, **validation (15%)**, and **test (15%)** datasets

The final cleaned and preprocessed review dataset is saved in:

```text
data/processed/
├── processed_reviews.csv
├── train.csv
├── validation.csv
└── test.csv
```

This dataset is used throughout the remainder of the project for exploratory data analysis, model training, and evaluation.

---

### Step 3: Exploratory Data Analysis

Run:

```bash
jupyter notebook notebooks/eda.ipynb
```

The notebook analyzes the processed dataset through visualizations and descriptive statistics, including:

- rating distribution
- sentiment distribution
- review length
- review volume by product
- additional analyses supporting the project's research questions

---

### Step 4: Train the Sentiment Classification Model

`src/train.py` is implemented and verified (fine-tunes `bert-base-uncased` on the train/validation splits, config-driven via `bert_config.yaml`). Training itself has not been run yet — results pending.

Run:

```bash
python src/train.py
```

---

### Step 5: Evaluate the Sentiment Classification Model

`src/evaluate.py` is implemented and verified (accuracy, macro/weighted/per-class F1, confusion matrix, classification report on the held-out test split). Results are pending until Step 4 has been run.

Run:

```bash
python src/evaluate.py
```

---

### Step 6: Generate Product Review Summaries

**To be completed.**

---

### Step 7: Evaluate Generated Summaries

**To be completed.**

---

## Documentation

The `docs/` directory contains supporting project documentation, including:

- **Project Proposal** – outlines the project objectives, research questions, implementation plan, and team responsibilities.
- **Model Research** – summarizes the literature review and background research conducted on the transformer models used in this project.
- **Model Framework** – describes the project's methodology, including model selection, data flow, variables, and how the models are integrated into the overall pipeline.

---

## Team Responsibilities

| Member | Role |
|--------|------|
| Sapna Patel | Dataset collection and preprocessing pipeline |
| Prema Parks | EDA, literature review, class-balancing experiments, GitHub structure |
| Yo Furusawa | BERT sentiment classification: tokenization, fine-tuning, evaluation |
| Jose Juan Enriquez Valdes | BART summarization and model pipeline (`src/model_runner.py`) |

---

## Milestones

| Milestone | Due | Status |
|-----------|-----|--------|
| M2: Project Proposal | Jul 5 | Done |
| M3: Data Pipeline | Jul 19 | In progress |
| M4: Model Pipeline | Jul 26 | Upcoming |
| M5: Final Submission | Aug 9 | Upcoming |
