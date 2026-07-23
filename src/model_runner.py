"""
BART model pipeline for Amazon Electronics review summarization.

The pipeline compares two summarization strategies:

1. Combined:
   Positive, neutral, and negative reviews are summarized together.

2. Sentiment-separated:
   Reviews are summarized separately by sentiment class.

The generated summaries and evaluation results are saved to CSV files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_processed_data
from evaluate import (
    create_strategy_comparison,
    evaluate_summaries,
    save_evaluation_outputs,
)
from summarizer import BartSummarizer
from utils import load_config


SENTIMENT_ORDER = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


def validate_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """Raise a clear error if required dataset columns are missing."""
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def select_sample_products(
    df: pd.DataFrame,
    product_column: str,
    sentiment_column: str,
    sample_products: int,
) -> list[str]:
    """
    Select products with strong sentiment coverage.

    Products containing all three sentiment classes are prioritized,
    followed by products with the highest number of reviews.
    """
    product_stats = (
        df.groupby(product_column)
        .agg(
            total_reviews=(product_column, "size"),
            sentiment_count=(sentiment_column, "nunique"),
        )
        .reset_index()
        .sort_values(
            by=["sentiment_count", "total_reviews"],
            ascending=[False, False],
        )
    )

    return (
        product_stats
        .head(sample_products)[product_column]
        .astype(str)
        .tolist()
    )


def select_reviews(
    group: pd.DataFrame,
    helpful_vote_column: str,
    maximum_reviews: int,
) -> pd.DataFrame:
    """
    Select the most useful reviews from a group.

    Reviews with more helpful votes are prioritized.
    """
    selected = group.copy()

    if helpful_vote_column in selected.columns:
        selected[helpful_vote_column] = pd.to_numeric(
            selected[helpful_vote_column],
            errors="coerce",
        ).fillna(0)

        selected = selected.sort_values(
            by=helpful_vote_column,
            ascending=False,
        )

    return selected.head(maximum_reviews)


def format_review(
    row: pd.Series,
    review_number: int,
    review_title_column: str,
    review_text_column: str,
    rating_column: str,
) -> str:
    """
    Format one review with clear boundaries for BART.

    Example:
        Review 1
        Rating: 5/5
        Title: Excellent headphones
        Text: Great battery life and sound quality.
    """
    title = str(
        row.get(review_title_column, "")
        if pd.notna(row.get(review_title_column, ""))
        else ""
    ).strip()

    text = str(
        row.get(review_text_column, "")
        if pd.notna(row.get(review_text_column, ""))
        else ""
    ).strip()

    rating_value = row.get(rating_column, "")
    rating_line = (
        f"Rating: {rating_value}/5" if pd.notna(rating_value) else "Rating: Unknown"
    )

    if not title and not text:
        return ""

    sections = [
        f"Review {review_number}",
        rating_line,
    ]

    if title:
        sections.append(f"Title: {title}")

    if text:
        sections.append(f"Text: {text}")

    return "\n".join(sections)


def truncate_to_word_limit(text: str, max_words: int) -> tuple[str, bool]:
    """
    Truncate text to at most max_words words, preferring line boundaries.

    Splits on "\\n" first so the "Review N / Rating / Title / Text" layout
    built by format_review (and the blank-line separators between reviews
    from format_review_group) survive truncation instead of being collapsed
    into one space-joined blob. Only splits mid-line when a single line's
    word count would overflow the remaining budget.

    Returns:
        Tuple of (possibly truncated text, whether truncation occurred).
    """
    lines = text.split("\n")
    line_words = [line.split() for line in lines]
    total_words = sum(len(words) for words in line_words)

    if total_words <= max_words:
        return text, False

    kept_lines = []
    remaining = max_words

    for line, words in zip(lines, line_words):
        if not words:
            if remaining > 0:
                kept_lines.append(line)
            continue

        if len(words) <= remaining:
            kept_lines.append(line)
            remaining -= len(words)
        elif remaining > 0:
            kept_lines.append(" ".join(words[:remaining]))
            remaining = 0
        else:
            break

    return "\n".join(kept_lines), True


def format_review_group(
    group: pd.DataFrame,
    cfg: dict,
) -> tuple[str, int]:
    """
    Select and format multiple reviews as one BART input document.

    Returns:
        Tuple of formatted text and number of usable reviews.
    """
    data_cfg = cfg["data"]

    selected = select_reviews(
        group=group,
        helpful_vote_column=data_cfg["helpful_vote_column"],
        maximum_reviews=int(data_cfg["reviews_per_group"]),
    )

    formatted_reviews = []

    for review_number, (_, row) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        formatted = format_review(
            row=row,
            review_number=review_number,
            review_title_column=data_cfg["review_title_column"],
            review_text_column=data_cfg["review_text_column"],
            rating_column=data_cfg["rating_column"],
        )

        if formatted:
            formatted_reviews.append(formatted)

    return "\n\n".join(formatted_reviews), len(formatted_reviews)


def prepare_summary_inputs(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Prepare combined and sentiment-separated summarization inputs.
    """
    data_cfg = cfg["data"]

    product_column = data_cfg["product_column"]
    title_column = data_cfg["product_title_column"]
    sentiment_column = data_cfg["sentiment_column"]

    required_columns = {
        product_column,
        title_column,
        sentiment_column,
        data_cfg["review_title_column"],
        data_cfg["review_text_column"],
        data_cfg["rating_column"],
    }

    validate_columns(df, required_columns)

    selected_products = select_sample_products(
        df=df,
        product_column=product_column,
        sentiment_column=sentiment_column,
        sample_products=int(data_cfg["sample_products"]),
    )

    # Word budget per sentiment section in the combined-strategy input (see
    # the comment below). Constant across products, read once here.
    max_words_per_section = int(data_cfg["max_words_per_sentiment_section"])

    output_rows = []

    for product_id in selected_products:
        product_group = df[
            df[product_column].astype(str) == str(product_id)
        ].copy()

        if product_group.empty:
            continue

        product_title = str(
            product_group[title_column]
            .fillna("Unknown product")
            .iloc[0]
        )

        # Combined strategy.
        #
        # Each sentiment section is capped to a fixed word budget before
        # concatenation. Without this, the combined document is truncated
        # by the tokenizer's default right-side truncation at generation
        # time -- since sections are always ordered negative/neutral/
        # positive, that silently drops the positive section first (and
        # often entirely) for any product with enough reviews to exceed
        # the model's input-token limit, systematically biasing combined
        # summaries toward negative/neutral content with no diagnostic.
        #
        # This word budget is only an approximation of the token budget
        # that actually matters (spec-heavy electronics review text can
        # run well above 1.3 tokens/word), so it reduces but doesn't
        # guarantee eliminating the underlying truncation risk. The actual
        # ground-truth check (comparing real token counts against
        # model.max_input_tokens, for both strategies) happens in
        # generate_summaries() below, right before generation.
        combined_sections = []
        combined_count = 0

        for sentiment in ["negative", "neutral", "positive"]:
            sentiment_group = product_group[
                product_group[sentiment_column] == sentiment
            ]

            formatted_text, review_count = format_review_group(
                sentiment_group,
                cfg,
            )

            if formatted_text:
                formatted_text, was_truncated = truncate_to_word_limit(
                    formatted_text,
                    max_words_per_section,
                )

                if was_truncated:
                    print(
                        f"Warning: {sentiment} section for product "
                        f"{product_id} exceeded {max_words_per_section} "
                        "words and was truncated so all sentiment "
                        "sections fit in the combined input."
                    )

                combined_sections.append(
                    f"{sentiment.upper()} REVIEWS\n{formatted_text}"
                )
                combined_count += review_count

        if combined_sections:
            output_rows.append(
                {
                    "parent_asin": str(product_id),
                    "product_title": product_title,
                    "strategy": "combined",
                    "sentiment": "all",
                    "review_count": combined_count,
                    "combined_reviews": "\n\n".join(combined_sections),
                }
            )

        # Sentiment-separated strategy.
        available_sentiments = sorted(
            product_group[sentiment_column]
            .dropna()
            .unique()
            .tolist(),
            key=lambda value: SENTIMENT_ORDER.get(str(value), 99),
        )

        for sentiment in available_sentiments:
            sentiment_group = product_group[
                product_group[sentiment_column] == sentiment
            ]

            formatted_text, review_count = format_review_group(
                sentiment_group,
                cfg,
            )

            if not formatted_text:
                continue

            output_rows.append(
                {
                    "parent_asin": str(product_id),
                    "product_title": product_title,
                    "strategy": "sentiment_separated",
                    "sentiment": str(sentiment),
                    "review_count": review_count,
                    "combined_reviews": formatted_text,
                }
            )

    prepared_df = pd.DataFrame(output_rows)

    if prepared_df.empty:
        raise ValueError(
            "No summary input documents were generated."
        )

    return prepared_df


def generate_summaries(
    prepared_df: pd.DataFrame,
    summarizer: BartSummarizer,
    cfg: dict,
) -> pd.DataFrame:
    """Generate BART summaries for all prepared documents."""
    model_cfg = cfg["model"]
    results = prepared_df.copy()

    summaries = []
    statuses = []
    errors = []

    print(f"\nGenerating {len(results)} summaries...\n")

    for position, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):
        print(
            f"[{position}/{len(results)}] "
            f"{row['product_title']} | "
            f"{row['strategy']} | "
            f"{row['sentiment']}"
        )

        max_input_tokens = int(model_cfg["max_input_tokens"])

        # Ground-truth truncation check using the real BART tokenizer, on
        # every row regardless of strategy. The combined-strategy word
        # budget above only approximates this and can undershoot on
        # spec/number-heavy review text, so this is the actual signal that
        # the tokenizer's own truncation (inside summarizer.summarize())
        # is about to silently drop content.
        true_token_count = summarizer.count_tokens(row["combined_reviews"])

        if true_token_count > max_input_tokens:
            print(
                f"Warning: input exceeds {max_input_tokens} tokens "
                f"({true_token_count} tokens) and will be truncated "
                "by the tokenizer at generation time."
            )

        try:
            summary = summarizer.summarize(
                text=row["combined_reviews"],
                max_input_tokens=max_input_tokens,
                max_length=int(
                    model_cfg["max_summary_length"]
                ),
                min_length=int(
                    model_cfg["min_summary_length"]
                ),
                num_beams=int(model_cfg["num_beams"]),
                length_penalty=float(
                    model_cfg["length_penalty"]
                ),
                no_repeat_ngram_size=int(
                    model_cfg["no_repeat_ngram_size"]
                ),
            )

            summaries.append(summary)
            statuses.append("success")
            errors.append("")

        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"

            print(f"Warning: {error_message}")

            summaries.append("")
            statuses.append("failed")
            errors.append(error_message)

    results["summary"] = summaries
    results["generation_status"] = statuses
    results["generation_error"] = errors

    return results


def save_summary_results(
    results_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Save generated summaries to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(path, index=False)

    print(f"\nSummaries saved to: {path}")


def main() -> None:
    """Run the full BART summarization and evaluation pipeline."""
    cfg = load_config()["bart"]

    print("Loading processed dataset...")

    reviews_df = load_processed_data(
        cfg["data"]["processed_path"]
    )

    print(f"Loaded {len(reviews_df):,} reviews.")

    prepared_df = prepare_summary_inputs(
        reviews_df,
        cfg,
    )

    print(
        f"Selected {prepared_df['parent_asin'].nunique()} products."
    )

    print(
        f"Prepared {len(prepared_df)} summary input documents."
    )

    print("\nPrepared input distribution:\n")

    print(
        prepared_df[
            [
                "parent_asin",
                "product_title",
                "strategy",
                "sentiment",
                "review_count",
            ]
        ].to_string(index=False)
    )

    summarizer = BartSummarizer(
        cfg["model"]["name"]
    )

    results_df = generate_summaries(
        prepared_df=prepared_df,
        summarizer=summarizer,
        cfg=cfg,
    )

    save_summary_results(
        results_df=results_df,
        output_path=cfg["outputs"]["summary_path"],
    )

    successful_df = results_df[
        results_df["generation_status"] == "success"
    ].copy()

    if successful_df.empty:
        print("No summaries were successfully generated.")
        return

    evaluation_df = evaluate_summaries(
        successful_df
    )

    comparison_df = create_strategy_comparison(
        evaluation_df
    )

    save_evaluation_outputs(
        evaluation_df=evaluation_df,
        comparison_df=comparison_df,
        evaluation_path=cfg["outputs"]["evaluation_path"],
        comparison_path=cfg["outputs"]["comparison_path"],
    )

    print("\nStrategy comparison:\n")
    print(comparison_df.to_string(index=False))

    failed_count = int(
        (results_df["generation_status"] == "failed").sum()
    )

    print("\nPipeline completed.")
    print(f"Successful summaries: {len(successful_df)}")
    print(f"Failed summaries: {failed_count}")


if __name__ == "__main__":
    main()