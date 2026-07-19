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
        ValueError: If any class in df["label"] has too few members to be
            stratified across all three splits.
    """
    class_counts = df["label"].value_counts()
    too_small = class_counts[class_counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Cannot create a stratified train/validation/test split: "
            f"label(s) {too_small.index.tolist()} have fewer than 3 rows "
            f"(counts: {too_small.to_dict()}). Increase the sample size or "
            "relax upstream filters (e.g. min_review_length) so every class "
            "has enough rows to appear in all three splits."
        )

    test_size = 1.0 - train_size - val_size

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=df["label"]
    )

    relative_val = val_size / (val_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        random_state=seed,
        stratify=temp_df["label"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )
