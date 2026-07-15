"""Load and split the processed reviews dataset."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


LABEL_MAP = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}


def assign_sentiment_label(rating: int) -> str:
    return LABEL_MAP[rating]


def load_processed_data(path: str = "data/processed_reviews.csv.gz") -> pd.DataFrame:
    df = pd.read_csv(path, compression="gzip")
    df["sentiment"] = df["rating"].map(assign_sentiment_label)
    df["label"] = df["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
    df["input_text"] = df["review_title"].fillna("") + " " + df["text"].fillna("")
    df["input_text"] = df["input_text"].str.strip()
    return df


def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_size = 1.0 - train_size - val_size
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), random_state=seed, stratify=df["label"]
    )
    relative_val = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - relative_val), random_state=seed, stratify=temp_df["label"]
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_processed_data()
    print(f"Total samples: {len(df):,}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}\n")
    train, val, test = split_data(df)
    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
