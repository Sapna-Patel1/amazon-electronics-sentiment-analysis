"""
Evaluate a fine-tuned BERT sentiment classifier on the held-out test set.

Loads the saved model from models/bert_sentiment/, runs inference on the
test split, and reports accuracy, per-class F1, confusion matrix, and a
full classification report. Results are saved to outputs/evaluation_results.txt.

Usage:
    python src/evaluate.py
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_loader import load_processed_data, split_data

LABEL_NAMES = ["negative", "neutral", "positive"]


def load_config(path: str = "configs/bert_config.yaml") -> dict:
    """Load evaluation configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_model(model_dir: str, test_df: pd.DataFrame, max_length: int = 256) -> dict:
    """Run inference on the test set and compute evaluation metrics.

    Args:
        model_dir: Path to the directory containing the saved model and tokenizer.
        test_df: DataFrame with 'input_text' and 'label' columns.
        max_length: Maximum token length for truncation.

    Returns:
        Dictionary containing:
        - ``accuracy``: Overall accuracy score.
        - ``f1_macro``: Macro-averaged F1 across all three classes.
        - ``f1_weighted``: Weighted-averaged F1.
        - ``f1_per_class``: List of per-class F1 scores [negative, neutral, positive].
        - ``classification_report``: Full sklearn classification report string.
        - ``confusion_matrix``: Confusion matrix as a nested list.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=tokenizer,
        truncation=True,
        max_length=max_length,
        device=0 if __import__("torch").cuda.is_available() else -1,
    )

    texts = test_df["input_text"].tolist()
    true_labels = test_df["label"].tolist()

    raw_preds = clf(texts, batch_size=32)
    pred_labels = [int(p["label"].split("_")[-1]) for p in raw_preds]

    report = classification_report(true_labels, pred_labels, target_names=LABEL_NAMES)
    cm = confusion_matrix(true_labels, pred_labels)

    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_macro": f1_score(true_labels, pred_labels, average="macro"),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted"),
        "f1_per_class": f1_score(true_labels, pred_labels, average=None).tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def main():
    """Load the trained model, evaluate on the test set, and save results."""
    cfg = load_config()
    df = load_processed_data(cfg["data"]["processed_path"])
    _, _, test_df = split_data(df, seed=cfg["data"]["seed"])

    model_dir = cfg["outputs"]["model_dir"]
    print(f"Evaluating model from {model_dir} on {len(test_df):,} test samples...")

    results = evaluate_model(model_dir, test_df, cfg["model"]["max_length"])

    print(f"\nAccuracy:    {results['accuracy']:.4f}")
    print(f"F1 Macro:    {results['f1_macro']:.4f}")
    print(f"F1 Weighted: {results['f1_weighted']:.4f}")
    print(f"\nPer-class F1: {dict(zip(LABEL_NAMES, results['f1_per_class']))}")
    print(f"\n{results['classification_report']}")

    output_path = Path(cfg["outputs"]["results_dir"]) / "evaluation_results.txt"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        f.write(results["classification_report"])
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
