"""Shared text-cleaning, label-mapping, and splitting logic for the data pipeline.

Used by both preprocess.py and data_loader.py so the two scripts don't
maintain independent copies of the same logic.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Mapping from star rating to integer label used by BERT (0=negative, 1=neutral, 2=positive)
RATING_TO_LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# Canonical label order, shared so train.py/evaluate.py don't each redeclare it
SENTIMENT_LABELS = [0, 1, 2]
LABEL_NAMES = ["negative", "neutral", "positive"]


def model_dir_for_variant(base_dir: str, use_class_weights: bool) -> str:
    """Build the per-variant model directory name for the RQ2 experiment.

    Shared by train.py (to save to) and evaluate.py (to load from) so the
    two can't drift apart into looking at different directory names.

    Args:
        base_dir: The configured base model directory (outputs.model_dir).
        use_class_weights: Whether this is the class-weighted variant.

    Returns:
        base_dir with "_weighted" or "_baseline" appended.
    """
    return base_dir + ("_weighted" if use_class_weights else "_baseline")


def get_sentiment(rating):
    """Convert a numeric star rating to a lowercase sentiment string.

    Args:
        rating: Integer star rating (1-5).

    Returns:
        "negative" for ratings below 3, "neutral" for 3, "positive" above 3.
    """
    if rating < 3:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def clean_text(series: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and collapse internal whitespace runs.

    Args:
        series: Series of raw text values.

    Returns:
        Series with whitespace normalized.
    """
    return series.str.strip().str.replace(r"\s+", " ", regex=True)


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train, validation, and test sets.

    Args:
        df: Full labeled DataFrame (must contain a 'label' column).
        train_size: Fraction of data for training (default 0.70).
        val_size: Fraction of data for validation (default 0.15).
            The test set receives the remaining fraction (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df), each reset-indexed.

    Raises:
        ValueError: If train_size + val_size >= 1.0 (leaves no room for a
            test split), or if any class in df["label"] is too small to be
            stratified across all three splits.
    """
    if train_size + val_size >= 1.0:
        raise ValueError(
            f"train_size ({train_size}) + val_size ({val_size}) must be "
            "less than 1.0 so a non-empty test split remains."
        )

    # No fixed per-class row-count threshold can catch every failure mode
    # up front: sklearn's proportional rounding across the two chained
    # stratified splits below can still round a class down to a single row
    # when multiple classes are simultaneously small, even if each one
    # individually clears a naive ">= N rows" guard (verified: a
    # 5-rows/5-rows/100-rows class split fails on every seed tested, despite
    # each small class clearing a ">= 5" check on its own). So instead of
    # guessing a "big enough" threshold, the actual split is attempted and
    # any resulting failure is caught and re-raised with the same clear
    # guidance, rather than letting sklearn's more opaque error surface
    # directly. The train_size/val_size guard above keeps this except
    # block scoped to genuine stratification failures, rather than also
    # catching (and mislabeling) an invalid split-ratio configuration.
    class_counts = df["label"].value_counts()

    def stratification_error(cause: Exception) -> ValueError:
        return ValueError(
            "Cannot create a stratified train/validation/test split: "
            "one or more classes are too small for this split ratio "
            f"(class counts: {class_counts.to_dict()}). Increase the "
            "sample size or relax upstream filters (e.g. min_review_length) "
            f"so every class has enough rows to appear in all three splits. "
            f"Underlying error: {cause}"
        )

    test_size = 1.0 - train_size - val_size

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            random_state=seed,
            stratify=df["label"]
        )
    except ValueError as exc:
        raise stratification_error(exc) from exc

    relative_val = val_size / (val_size + test_size)

    try:
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - relative_val),
            random_state=seed,
            stratify=temp_df["label"]
        )
    except ValueError as exc:
        raise stratification_error(exc) from exc

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
