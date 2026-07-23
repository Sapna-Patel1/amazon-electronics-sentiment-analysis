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

import re
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

REVIEW_HEADER_PATTERN = re.compile(r"^Review \d+$", re.MULTILINE)


def count_surviving_reviews(formatted_text: str) -> int:
    """
    Count how many reviews are actually present in (possibly truncated)
    formatted text, by counting intact "Review N" header lines.

    Counting reviews before truncate_to_word_limit() runs would overstate
    what survives, since truncation can drop one or more whole reviews
    from the end of a section -- so this should always be called on the
    (possibly truncated) text, not on a count computed earlier.
    """
    return len(REVIEW_HEADER_PATTERN.findall(formatted_text))


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
) -> str:
    """
    Select and format multiple reviews as one BART input document.

    Callers should use count_surviving_reviews() on the (possibly
    truncated) result rather than counting reviews here, since truncation
    can drop whole reviews from the end of the text.
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

    return "\n\n".join(formatted_reviews)


# Worst-case tokens-per-word ratio, calibrated against real committed
# data rather than a synthetic adversarial string (an artificially
# spec-dense test string measured ~2.13 tokens/word, but that doesn't
# reflect real review text composition): the actual max ratio observed
# across real generated combined-strategy and sentiment-separated rows
# was ~1.46-1.51. Used only as a sanity check that the configured word
# budgets below are still plausible relative to model.max_input_tokens --
# not used for the actual truncation, which always operates on real token
# counts (see generate_summaries()'s ground-truth check).
WORST_CASE_TOKENS_PER_WORD = 1.5


def warn_if_budgets_inconsistent(
    max_input_tokens: int,
    max_words_per_section: int,
    max_words_per_separated_group: int,
) -> None:
    """
    Warn if the configured word budgets look inconsistent with
    model.max_input_tokens, so a future change to max_input_tokens (e.g.
    switching to a model with a different context window) doesn't leave
    these two independently-chosen word budgets silently stale.
    """
    worst_case_combined_tokens = (
        3 * max_words_per_section * WORST_CASE_TOKENS_PER_WORD
    )
    worst_case_separated_tokens = (
        max_words_per_separated_group * WORST_CASE_TOKENS_PER_WORD
    )

    if worst_case_combined_tokens > max_input_tokens:
        print(
            f"Warning: max_words_per_sentiment_section ({max_words_per_section}) "
            f"x 3 sections at a worst-case {WORST_CASE_TOKENS_PER_WORD} "
            f"tokens/word ({worst_case_combined_tokens:.0f} tokens) exceeds "
            f"model.max_input_tokens ({max_input_tokens}) -- consider "
            "lowering max_words_per_sentiment_section."
        )

    if worst_case_separated_tokens > max_input_tokens:
        print(
            f"Warning: max_words_per_separated_group "
            f"({max_words_per_separated_group}) at a worst-case "
            f"{WORST_CASE_TOKENS_PER_WORD} tokens/word "
            f"({worst_case_separated_tokens:.0f} tokens) exceeds "
            f"model.max_input_tokens ({max_input_tokens}) -- consider "
            "lowering max_words_per_separated_group."
        )


def prepare_group_text(
    group: pd.DataFrame,
    cfg: dict,
    max_words: int,
    context: str,
) -> tuple[str, int]:
    """
    Format a review group and cap it to a word budget.

    Shared by both the combined strategy's per-sentiment sections and the
    sentiment-separated strategy's single section, so the format ->
    truncate -> warn -> count sequence isn't duplicated across the two
    call sites in prepare_summary_inputs() and can't drift between them.

    Args:
        group: Reviews to format (already filtered to one sentiment).
        cfg: Full config dict (cfg["data"] is used by format_review_group).
        max_words: Word budget to truncate to.
        context: Description used in the truncation warning, e.g.
            "negative section for product B01234ABCD".

    Returns:
        Tuple of (possibly truncated formatted text, number of reviews
        surviving truncation). Text is "" and count is 0 if the group had
        no usable reviews.
    """
    formatted_text = format_review_group(group, cfg)

    if not formatted_text:
        return "", 0

    formatted_text, was_truncated = truncate_to_word_limit(
        formatted_text,
        max_words,
    )

    if was_truncated:
        print(f"Warning: {context} exceeded {max_words} words and was truncated.")

    return formatted_text, count_surviving_reviews(formatted_text)


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

    # Word budgets (see comments below). Constant across products, read once here.
    max_words_per_section = int(data_cfg["max_words_per_sentiment_section"])
    max_words_per_separated_group = int(data_cfg["max_words_per_separated_group"])
    warn_if_budgets_inconsistent(
        max_input_tokens=int(cfg["model"]["max_input_tokens"]),
        max_words_per_section=max_words_per_section,
        max_words_per_separated_group=max_words_per_separated_group,
    )

    # Cast once and reuse across every iteration below, instead of
    # recomputing a full-column string cast (over the whole dataset) once
    # per selected product.
    product_column_as_str = df[product_column].astype(str)

    output_rows = []

    for product_id in selected_products:
        product_group = df[
            product_column_as_str == str(product_id)
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

            formatted_text, review_count = prepare_group_text(
                sentiment_group,
                cfg,
                max_words_per_section,
                context=f"{sentiment} section for product {product_id}",
            )

            if formatted_text:
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

            formatted_text, review_count = prepare_group_text(
                sentiment_group,
                cfg,
                max_words_per_separated_group,
                context=(
                    f"{sentiment} sentiment-separated input for product "
                    f"{product_id}"
                ),
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