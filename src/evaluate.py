"""
Evaluate a fine-tuned BERT sentiment classifier on the held-out test set.

Loads the saved model from models/bert_sentiment_{baseline,weighted}/
(whichever variant configs/bert_config.yaml's training.use_class_weights
currently selects), runs inference on the test split, and reports
accuracy, precision, recall, F1 (macro/weighted/per-class), confusion
matrix, and a full classification report. Results are saved to
outputs/bert_evaluation_{baseline,weighted}.{txt,json}, and a random
10-row sample of predictions (review text, actual sentiment, predicted
sentiment) is saved to outputs/bert_sample_predictions_{baseline,weighted}.csv.

This module covers BERT evaluation only. BART summary evaluation (ROUGE,
compression ratio, lexical coverage, sentiment alignment, and manual
qualitative scoring) lives in summary_evaluation.py and is invoked directly
from model_runner.py rather than from here.

Usage:
    python src/evaluate.py
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from data_loader import load_processed_data
from utils import load_config
from utils.helpers import LABEL_NAMES, SENTIMENT_LABELS, model_dir_for_variant


def evaluate_model(
    model_dir: str,
    test_df: pd.DataFrame,
    max_length: int = 256,
    batch_size: int = 16,
) -> dict:
    """Run inference on the test set and compute evaluation metrics.

    Args:
        model_dir: Path to the directory containing the saved model and tokenizer.
        test_df: DataFrame with 'input_text' and 'label' columns.
        max_length: Maximum token length for truncation.
        batch_size: Batch size for pipeline inference.

    Returns:
        Dictionary containing:
        - ``accuracy``: Overall accuracy score.
        - ``precision_macro``: Macro-averaged precision across all three classes.
        - ``recall_macro``: Macro-averaged recall across all three classes.
        - ``f1_macro``: Macro-averaged F1 across all three classes.
        - ``f1_weighted``: Weighted-averaged F1.
        - ``precision_per_class``: List of per-class precision [negative, neutral, positive].
        - ``recall_per_class``: List of per-class recall [negative, neutral, positive].
        - ``f1_per_class``: List of per-class F1 scores [negative, neutral, positive].
        - ``classification_report``: Full sklearn classification report string.
        - ``confusion_matrix``: Confusion matrix as a nested list.
        - ``pred_labels``: Predicted integer labels, aligned index-for-index
          with the input test_df.
        - ``tokenizer``: The tokenizer used for inference (for callers that
          need to reproduce exactly what the model saw, e.g. truncated text).

    Raises:
        FileNotFoundError: If model_dir doesn't exist (e.g. train.py hasn't
            been run yet).
    """
    if not Path(model_dir).is_dir():
        raise FileNotFoundError(
            f"Model directory '{model_dir}' does not exist. "
            "Run `python src/train.py` first to produce a trained model."
        )

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = -1

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    clf = pipeline(
        "text-classification",
        model=model_dir,
        tokenizer=tokenizer,
        truncation=True,
        max_length=max_length,
        device=device,
    )

    texts = test_df["input_text"].tolist()
    true_labels = test_df["label"].tolist()

    raw_preds = clf(texts, batch_size=batch_size)
    pred_labels = [int(p["label"].split("_")[-1]) for p in raw_preds]

    report = classification_report(
        true_labels, pred_labels, labels=SENTIMENT_LABELS, target_names=LABEL_NAMES
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=SENTIMENT_LABELS)

    # zero_division=np.nan (rather than the sklearn default of silently
    # returning 0.0) so a class that was never predicted is distinguishable
    # from a class that was predicted but always wrong -- both would
    # otherwise report an identical, misleadingly-precise 0.0.
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=SENTIMENT_LABELS, average=None, zero_division=np.nan
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=SENTIMENT_LABELS, average="macro", zero_division=np.nan
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=SENTIMENT_LABELS, average="weighted", zero_division=np.nan
    )

    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "f1_per_class": f1_per_class.tolist(),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "pred_labels": pred_labels,
        "tokenizer": tokenizer,
    }


def save_sample_predictions(
    test_df: pd.DataFrame,
    pred_labels: list,
    tokenizer,
    max_length: int,
    output_path: Path,
    n: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Save a random sample of test-set predictions for manual inspection.

    review_text is truncated to the same max_length the model actually saw
    during inference (rather than the full raw text), so a reviewer isn't
    misled by content past the truncation point when judging a prediction.

    Args:
        test_df: DataFrame with 'input_text' and 'label' columns (test split).
        pred_labels: Predicted integer labels, aligned index-for-index with test_df.
        tokenizer: Tokenizer used for inference, for truncating review_text consistently.
        max_length: Maximum token length used during inference.
        output_path: CSV path to write the sample to.
        n: Number of rows to sample.
        seed: Random seed for reproducible sampling.

    Returns:
        The sampled DataFrame that was written to output_path.
    """
    sample = test_df.sample(n=min(n, len(test_df)), random_state=seed)
    sample_pred_labels = np.array(pred_labels)[sample.index]

    truncated_text = [
        tokenizer.decode(
            tokenizer(text, truncation=True, max_length=max_length)["input_ids"],
            skip_special_tokens=True,
        )
        for text in sample["input_text"]
    ]

    output = pd.DataFrame(
        {
            "review_text": truncated_text,
            "actual_sentiment": [LABEL_NAMES[label] for label in sample["label"]],
            "predicted_sentiment": [LABEL_NAMES[label] for label in sample_pred_labels],
        }
    )
    output.to_csv(output_path, index=False)
    return output


def main():
    """Load the trained model, evaluate on the test set, and save results."""
    cfg = load_config()
    print(f"Loading test split from {cfg['data']['test_path']}...")
    test_df = load_processed_data(cfg["data"]["test_path"])

    use_class_weights = cfg["training"].get("use_class_weights", False)
    variant = "weighted" if use_class_weights else "baseline"
    model_dir = model_dir_for_variant(cfg["outputs"]["model_dir"], use_class_weights)
    print(f"Evaluating model from {model_dir} on {len(test_df):,} test samples...")

    results = evaluate_model(
        model_dir,
        test_df,
        max_length=cfg["model"]["max_length"],
        batch_size=cfg["training"]["batch_size"],
    )

    print(f"\nAccuracy:         {results['accuracy']:.4f}")
    print(f"Precision Macro:  {results['precision_macro']:.4f}")
    print(f"Recall Macro:     {results['recall_macro']:.4f}")
    print(f"F1 Macro:         {results['f1_macro']:.4f}")
    print(f"F1 Weighted:      {results['f1_weighted']:.4f}")
    print(f"\nPer-class precision: {dict(zip(LABEL_NAMES, results['precision_per_class']))}")
    print(f"Per-class recall:    {dict(zip(LABEL_NAMES, results['recall_per_class']))}")
    print(f"Per-class F1:        {dict(zip(LABEL_NAMES, results['f1_per_class']))}")
    print(f"\n{results['classification_report']}")

    results_dir = Path(cfg["outputs"]["results_dir"])
    results_dir.mkdir(exist_ok=True)

    text_path = results_dir / f"bert_evaluation_{variant}.txt"
    with open(text_path, "w") as f:
        f.write(results["classification_report"])
    print(f"Results saved to {text_path}")

    json_path = results_dir / f"bert_evaluation_{variant}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "variant": variant,
                "accuracy": results["accuracy"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "f1_macro": results["f1_macro"],
                "f1_weighted": results["f1_weighted"],
                "precision_per_class": dict(zip(LABEL_NAMES, results["precision_per_class"])),
                "recall_per_class": dict(zip(LABEL_NAMES, results["recall_per_class"])),
                "f1_per_class": dict(zip(LABEL_NAMES, results["f1_per_class"])),
                "confusion_matrix": results["confusion_matrix"],
            },
            f,
            indent=2,
        )
    print(f"Structured results saved to {json_path}")

    samples_path = results_dir / f"bert_sample_predictions_{variant}.csv"
    save_sample_predictions(
        test_df,
        results["pred_labels"],
        results["tokenizer"],
        cfg["model"]["max_length"],
        samples_path,
        n=cfg["outputs"].get("num_sample_predictions", 10),
        seed=cfg["training"]["seed"],
    )
    print(f"Sample predictions saved to {samples_path}")


if __name__ == "__main__":
    main()
