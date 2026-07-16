"""
Evaluation utilities for BART-generated review summaries.

Because the Amazon dataset does not contain human-written reference
summaries, ROUGE is calculated against the source review text and is
reported as a source-coverage diagnostic, not as reference-summary ROUGE.

Human evaluation remains necessary for relevance, coherence,
conciseness, and pros/cons coverage.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rouge_score import rouge_scorer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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


def lexical_coverage(source: str, summary: str) -> float:
    """Measure how much summary vocabulary appears in the source."""
    source_words = set(content_tokens(source))
    summary_words = set(content_tokens(summary))

    if not summary_words:
        return 0.0

    return len(source_words & summary_words) / len(summary_words)


def novelty_ratio(source: str, summary: str) -> float:
    """Measure summary vocabulary not copied from the source."""
    source_words = set(content_tokens(source))
    summary_words = set(content_tokens(summary))

    if not summary_words:
        return 0.0

    return len(summary_words - source_words) / len(summary_words)


def repetition_ratio(summary: str) -> float:
    """Measure repeated content words in the summary."""
    words = content_tokens(summary)

    if not words:
        return 0.0

    counts = Counter(words)

    repeated = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated / len(words)


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
            lexical_coverage(source, summary)
        )

        novelty_ratios.append(
            novelty_ratio(source, summary)
        )

        repetition_ratios.append(
            repetition_ratio(summary)
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


def parse_args() -> argparse.Namespace:
    """Read standalone evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated BART summaries."
    )

    parser.add_argument(
        "--input",
        default="outputs/summary_samples.csv",
    )

    parser.add_argument(
        "--evaluation-output",
        default="outputs/summary_evaluation.csv",
    )

    parser.add_argument(
        "--comparison-output",
        default="outputs/strategy_comparison.csv",
    )

    return parser.parse_args()


def main() -> None:
    """Run evaluation independently from model generation."""
    args = parse_args()

    results_df = pd.read_csv(args.input)

    evaluation_df = evaluate_summaries(
        results_df
    )

    comparison_df = create_strategy_comparison(
        evaluation_df
    )

    save_evaluation_outputs(
        evaluation_df=evaluation_df,
        comparison_df=comparison_df,
        evaluation_path=args.evaluation_output,
        comparison_path=args.comparison_output,
    )

    print("\nStrategy comparison:\n")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()