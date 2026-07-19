# The file preprocesses the Amazon Electronics dataset by merging the reviews and metadata datasets, cleaning the data, creating sentiment labels, and splitting the data into training, validation, and testing datasets.

# Import the pandas library for data manipulation
import pandas as pd

from utils import load_config
from utils.helpers import RATING_TO_LABEL, clean_text, get_sentiment, split_data


def preprocess_data():
    """Merge, clean, label, and split the raw review and metadata samples.

    Reads the sample CSVs produced by data_loader.py, merges them on
    parent_asin, removes duplicates and rows with missing or too-short review
    text, creates sentiment and integer label columns, builds the input_text
    field used by BERT, and saves the preprocessed dataset plus train/val/test
    splits to data/processed/.
    """
    cfg = load_config("configs/data_config.yaml")
    split_cfg = cfg["split"]
    min_review_length = cfg["preprocessing"]["min_review_length"]
    seed = cfg["seed"]

    # Load the already-filtered sample datasets
    metadata_df = pd.read_csv("data/raw/metadata_sample.csv")
    reviews_df = pd.read_csv("data/raw/reviews_sample.csv")

    # Merge the reviews and metadata datasets
    merged_df = pd.merge(
        reviews_df,
        metadata_df,
        on="parent_asin",
        how="inner"
    )

    # Check for missing values
    print("\nMissing Values")
    print("-" * 30)
    print(merged_df.isnull().sum())

    # Remove reviews with missing review text
    merged_df = merged_df.dropna(subset=["text"])

    # Replace missing review titles with an empty string
    merged_df["review_title"] = merged_df["review_title"].fillna("")

    # Replace missing product categories with "Unknown"
    merged_df["main_category"] = merged_df["main_category"].fillna("Unknown")

    # Count duplicate rows
    duplicate_count = merged_df.duplicated().sum()

    print("\nDuplicate Rows")
    print("-" * 30)
    print(f"Number of duplicate rows: {duplicate_count}")

    # Remove duplicate rows
    merged_df = merged_df.drop_duplicates().reset_index(drop=True)

    # Print the new number of rows
    print(f"\nRows after removing duplicates: {len(merged_df):,}")

    # Clean the review title and review text by removing extra spaces
    merged_df["review_title"] = clean_text(merged_df["review_title"])
    merged_df["text"] = clean_text(merged_df["text"])

    # Drop reviews whose body is too short to carry meaningful sentiment content
    rows_before_length_filter = len(merged_df)
    merged_df = merged_df[merged_df["text"].str.len() >= min_review_length].reset_index(drop=True)

    print("\nShort Reviews")
    print("-" * 30)
    print(f"Rows dropped for text shorter than {min_review_length} characters: "
          f"{rows_before_length_filter - len(merged_df):,}")

    # Combine the review title and review text into one column.
    # When the title is empty, omit the separator so the result is not ". text".
    merged_df["review"] = merged_df.apply(
        lambda row: row["text"] if row["review_title"] == ""
        else row["review_title"] + ". " + row["text"],
        axis=1,
    )

    # Remove extra spaces after combining the text
    merged_df["review"] = clean_text(merged_df["review"])

    # Create the lowercase sentiment column (negative / neutral / positive)
    merged_df["sentiment"] = merged_df["rating"].apply(get_sentiment)

    # Create the integer label column required by BERT (0=negative, 1=neutral, 2=positive)
    merged_df["label"] = merged_df["rating"].map(RATING_TO_LABEL)

    # Create the input_text column used by train.py (title + space + body)
    merged_df["input_text"] = (
        merged_df["review_title"].fillna("") + " " + merged_df["text"].fillna("")
    ).str.strip()

    # Check the sentiment distribution
    print("\nSentiment Distribution")
    print("-" * 30)
    print(merged_df["sentiment"].value_counts())

    # Print final dataset information
    print("\nFinal Dataset")
    print("-" * 30)
    print(f"Number of Rows: {len(merged_df):,}")
    print(f"Number of Columns: {len(merged_df.columns)}")

    # Review the first 5 rows
    print("\nFirst 5 Rows")
    print("-" * 30)
    print(merged_df.head())

    # Save the cleaned dataset to the processed folder
    merged_df.to_csv(
    "data/processed/preprocessed_reviews.csv",
    index=False
    )

    # Confirm that the file was saved successfully
    print("\nThe cleaned dataset was saved successfully.")
    print("Location: data/processed/preprocessed_reviews.csv")

    # Split the cleaned dataset into training, validation, and testing sets
    train_df, val_df, test_df = split_data(
        merged_df,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        seed=seed,
    )

    print("\nDataset Split")
    print("-" * 30)
    print(f"Training Set: {len(train_df):,} rows")
    print(f"Validation Set: {len(val_df):,} rows")
    print(f"Testing Set: {len(test_df):,} rows")

    # Save the training dataset
    train_df.to_csv(
    "data/processed/train.csv",
    index=False
    )

    # Save the validation dataset
    val_df.to_csv(
    "data/processed/validation.csv",
    index=False
    )

    # Save the testing dataset
    test_df.to_csv(
    "data/processed/test.csv",
    index=False
    )

    # Confirm that the datasets were saved successfully
    print("\nTraining, validation, and testing datasets were saved successfully.")
    print("Location: data/processed/")


if __name__ == "__main__":
    preprocess_data()
