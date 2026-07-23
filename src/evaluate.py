"""
Evaluation metrics for both the BERT sentiment classifier and BART-generated
review summaries.

BERT: loads the saved model from models/bert_sentiment_{baseline,weighted}/
(whichever variant configs/model_config.yaml's bert.training.use_class_weights
currently selects), runs inference on the test split, and reports
accuracy, precision, recall, F1 (macro/weighted/per-class), confusion
matrix, and a full classification report. Results are saved to
outputs/bert_evaluation_{baseline,weighted}.{txt,json}, and a random
10-row sample of predictions (review text, actual sentiment, predicted
sentiment) is saved to outputs/bert_sample_predictions_{baseline,weighted}.csv.

BART: evaluate_summaries(), create_strategy_comparison(), and
save_evaluation_outputs() compute automatic diagnostics (ROUGE source
coverage, compression ratio, lexical coverage, novelty ratio, repetition
ratio, sentiment alignment) and blank manual-qualitative-scoring fields for
BART-generated summaries. These are invoked from model_runner.py as part of
the end-to-end BART pipeline rather than run standalone from this file.

Usage:
    python src/evaluate.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import torch
import numpy as np
import pandas as pd
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


# --- BART summary evaluation --------------------------------------------
#
# Because the Amazon dataset does not contain human-written reference
# summaries, ROUGE is calculated against the source review text and is
# reported as a source-coverage diagnostic, not as reference-summary ROUGE.
# Human evaluation remains necessary for relevance, coherence, conciseness,
# and pros/cons coverage.

WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "for", "from", "had", "has", "have", "he", "her",
    "his", "i", "in", "is", "it", "its", "of", "on", "or",
    "our", "she", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return WORD_PATTERN.findall(str(text).lower())


def content_tokens(text: str) -> list[str]:
    """Return non-stopword tokens."""
    return [
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) > 1
    ]


def lexical_coverage(source_words: set[str], summary_words: set[str]) -> float:
    """Measure how much summary vocabulary appears in the source.

    Args:
        source_words: Content tokens (see content_tokens()) from the source text.
        summary_words: Content tokens from the summary text.
    """
    if not summary_words:
        return 0.0

    return len(source_words & summary_words) / len(summary_words)


def novelty_ratio(source_words: set[str], summary_words: set[str]) -> float:
    """Measure summary vocabulary not copied from the source.

    Args:
        source_words: Content tokens (see content_tokens()) from the source text.
        summary_words: Content tokens from the summary text.
    """
    if not summary_words:
        return 0.0

    return len(summary_words - source_words) / len(summary_words)


def repetition_ratio(summary_words: list[str]) -> float:
    """Measure repeated content words in the summary.

    Args:
        summary_words: Content tokens (see content_tokens()) from the summary text.
    """
    if not summary_words:
        return 0.0

    counts = Counter(summary_words)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / len(summary_words)


def sentiment_alignment(
    expected_sentiment: str,
    compound_score: float,
) -> float | None:
    """Compare expected group sentiment with VADER summary sentiment."""
    expected = str(expected_sentiment).lower()

    if expected == "all":
        return None

    if compound_score >= 0.05:
        predicted = "positive"
    elif compound_score <= -0.05:
        predicted = "negative"
    else:
        predicted = "neutral"

    if predicted == expected:
        return 1.0

    if predicted == "neutral" or expected == "neutral":
        return 0.5

    return 0.0


def evaluate_summaries(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate automatic diagnostics and create human-score fields."""
    # Imported here rather than at module level so the BERT-only entry
    # point (evaluate_model()/main()) doesn't pay for loading rouge_score
    # (which transitively imports nltk) and vaderSentiment on every
    # `python src/evaluate.py` run -- only this BART-side function needs them.
    from rouge_score import rouge_scorer
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    required_columns = {
        "strategy",
        "sentiment",
        "combined_reviews",
        "summary",
    }

    missing = required_columns - set(results_df.columns)

    if missing:
        raise ValueError(
            f"Missing evaluation columns: {sorted(missing)}"
        )

    analyzer = SentimentIntensityAnalyzer()

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    evaluated = results_df.copy()

    source_counts = []
    summary_counts = []
    compression_ratios = []
    lexical_coverages = []
    novelty_ratios = []
    repetition_ratios = []
    source_sentiments = []
    summary_sentiments = []
    sentiment_alignments = []
    rouge1_coverages = []
    rouge2_coverages = []
    rougeL_coverages = []

    for _, row in evaluated.iterrows():
        source = str(row["combined_reviews"])
        summary = str(row["summary"])

        source_count = len(tokenize(source))
        summary_count = len(tokenize(summary))

        # Computed once and reused across lexical_coverage/novelty_ratio/
        # repetition_ratio below, rather than each of those independently
        # re-running content_tokens() (regex tokenize + stopword filter)
        # on the same source/summary strings.
        summary_content_words = content_tokens(summary)
        source_word_set = set(content_tokens(source))
        summary_word_set = set(summary_content_words)

        source_compound = analyzer.polarity_scores(
            source
        )["compound"]

        summary_compound = analyzer.polarity_scores(
            summary
        )["compound"]

        rouge_scores = rouge.score(
            source,
            summary,
        )

        source_counts.append(source_count)
        summary_counts.append(summary_count)

        compression_ratios.append(
            summary_count / source_count
            if source_count
            else 0.0
        )

        lexical_coverages.append(
            lexical_coverage(source_word_set, summary_word_set)
        )

        novelty_ratios.append(
            novelty_ratio(source_word_set, summary_word_set)
        )

        repetition_ratios.append(
            repetition_ratio(summary_content_words)
        )

        source_sentiments.append(source_compound)
        summary_sentiments.append(summary_compound)

        sentiment_alignments.append(
            sentiment_alignment(
                row["sentiment"],
                summary_compound,
            )
        )

        # Recall is used because this is source-coverage evaluation.
        rouge1_coverages.append(
            rouge_scores["rouge1"].recall
        )

        rouge2_coverages.append(
            rouge_scores["rouge2"].recall
        )

        rougeL_coverages.append(
            rouge_scores["rougeL"].recall
        )

    evaluated["source_word_count"] = source_counts
    evaluated["summary_word_count"] = summary_counts
    evaluated["compression_ratio"] = compression_ratios
    evaluated["lexical_coverage"] = lexical_coverages
    evaluated["novelty_ratio"] = novelty_ratios
    evaluated["repetition_ratio"] = repetition_ratios
    evaluated["source_vader_compound"] = source_sentiments
    evaluated["summary_vader_compound"] = summary_sentiments
    evaluated["sentiment_alignment"] = sentiment_alignments

    evaluated["rouge1_source_coverage"] = rouge1_coverages
    evaluated["rouge2_source_coverage"] = rouge2_coverages
    evaluated["rougeL_source_coverage"] = rougeL_coverages

    # Human evaluators enter scores from 1 to 5.
    evaluated["manual_relevance_1_to_5"] = pd.NA
    evaluated["manual_coherence_1_to_5"] = pd.NA
    evaluated["manual_conciseness_1_to_5"] = pd.NA
    evaluated["manual_pros_cons_coverage_1_to_5"] = pd.NA
    evaluated["manual_notes"] = ""

    return evaluated


def create_strategy_comparison(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create a presentation-ready comparison by strategy."""
    metric_columns = [
        "source_word_count",
        "summary_word_count",
        "compression_ratio",
        "lexical_coverage",
        "novelty_ratio",
        "repetition_ratio",
        "sentiment_alignment",
        "rouge1_source_coverage",
        "rouge2_source_coverage",
        "rougeL_source_coverage",
        "manual_relevance_1_to_5",
        "manual_coherence_1_to_5",
        "manual_conciseness_1_to_5",
        "manual_pros_cons_coverage_1_to_5",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in evaluation_df.columns
    ]

    comparison = (
        evaluation_df
        .groupby("strategy", dropna=False)[available_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )

    counts = (
        evaluation_df
        .groupby("strategy")
        .size()
        .rename("summary_count")
        .reset_index()
    )

    return counts.merge(
        comparison,
        on="strategy",
        how="left",
    )


def save_evaluation_outputs(
    evaluation_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    evaluation_path: str,
    comparison_path: str,
) -> None:
    """Save detailed and strategy-level results."""
    evaluation_file = Path(evaluation_path)
    comparison_file = Path(comparison_path)

    evaluation_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_csv(
        evaluation_file,
        index=False,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    print(f"Evaluation saved to: {evaluation_file}")
    print(f"Strategy comparison saved to: {comparison_file}")


def main():
    """Load the trained model, evaluate on the test set, and save results."""
    cfg = load_config()["bert"]
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
