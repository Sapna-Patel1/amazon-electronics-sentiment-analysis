# Amazon Electronics Sentiment Analysis & Review Summarization

**Group 10** | Northeastern University  
Sapna Patel · Prema Parks · Yo Furusawa · Jose Juan Enriquez Valdes

---

## Project Overview

This project builds a two-part NLP pipeline on Amazon Electronics reviews:

1. **Sentiment Classification** — Fine-tune `bert-base-uncased` to classify reviews as positive, neutral, or negative using star-rating-derived labels.
2. **Review Summarization** — Use `facebook/bart-large-cnn` to generate abstractive summaries of grouped product reviews, comparing sentiment-separated vs. all-reviews-combined input strategies.

**Dataset:** [Amazon Reviews 2023 — Electronics](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) (McAuley Lab, UC San Diego). Full dataset: ~18.3M reviews across ~1.6M products. Working subset: ~276,750 reviews across ~10,000 products.

---

## Repository Structure

```
amazon-electronics-sentiment-analysis/
├── data/
│   ├── preprocessing.ipynb       # Data loading, cleaning, merging
│   └── processed_reviews.csv.gz  # Cleaned subset (~276k reviews)
├── experiments/
│   └── eda.ipynb                 # Exploratory data analysis
├── models/                       # Saved model checkpoints (git-ignored)
├── src/
│   ├── data_loader.py            # Load processed data, assign labels, split
│   ├── train.py                  # BERT fine-tuning script
│   └── evaluate.py               # Evaluation: accuracy, F1, confusion matrix
├── configs/
│   └── bert_config.yaml          # Hyperparameters and paths
├── outputs/                      # Generated summaries and evaluation results
├── docs/                         # Literature review and model documentation
├── requirements.txt
└── README.md
```

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

## Usage

### Data preprocessing

Open and run `data/preprocessing.ipynb` to reproduce `processed_reviews.csv.gz`.

### Exploratory data analysis

Open and run `experiments/eda.ipynb` for class distribution, review length analysis, and sentiment breakdowns.

### BERT fine-tuning

```bash
cd src
python train.py
```

Hyperparameters are in `configs/bert_config.yaml`. The trained model is saved to `models/bert_sentiment/`.

### Evaluation

```bash
cd src
python evaluate.py
```

Results (classification report, per-class F1) are saved to `outputs/evaluation_results.txt`.

---

## Sentiment Labels

| Star Rating | Label    | Numeric |
|-------------|----------|---------|
| 1–2 stars   | negative | 0       |
| 3 stars     | neutral  | 1       |
| 4–5 stars   | positive | 2       |

---

## Research Questions

- **RQ1:** How accurately can fine-tuned BERT classify Amazon Electronics reviews into positive, neutral, and negative?
- **RQ2:** How does class balancing affect BERT performance, especially neutral-class recall and F1?
- **RQ3:** Does sentiment-separated input improve BART-generated summary quality vs. all-reviews-combined input?

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
