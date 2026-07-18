# The file preprocesses the Amazon Electronics dataset by merging the reviews and metadata datasets, cleaning the data, creating sentiment labels, and splitting the data into training, validation, and testing datasets.

# Import the pandas library for data manipulation
import pandas as pd
# Splitting the data into test and validation sets
from sklearn.model_selection import train_test_split

# Mapping from star rating to integer label used by BERT (0=negative, 1=neutral, 2=positive)
RATING_TO_LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


def get_sentiment(rating):
    """Convert a numeric star rating to a lowercase sentiment string.

    Args:
        rating: Integer star rating (1-5).

    Returns:
        "negative" for ratings below 3, "neutral" for 3, "positive" above 3.
    """
    # Ratings below 3 are negative
    if rating < 3:
        return "negative"

    # A rating of 3 is neutral
    elif rating == 3:
        return "neutral"

    # Ratings above 3 are positive
    else:
        return "positive"


# Split the dataset into training, validation, and testing sets
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
    """
    # Calculate the percentage of data for the testing set
    test_size = 1.0 - train_size - val_size

    # Split the data into training and temporary datasets
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=df["label"]
    )

    # Calculate the validation size relative to the temporary dataset
    relative_val = val_size / (val_size + test_size)

    # Split the temporary dataset into validation and testing datasets
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

def preprocess_data():
    """Merge, clean, label, and split the raw review and metadata samples.

    Reads the sample CSVs produced by data_loader.py, merges them on
    parent_asin, removes duplicates and rows with missing review text,
    creates sentiment and integer label columns, builds the input_text field
    used by BERT, and saves the preprocessed dataset plus train/val/test
    splits to data/processed/.
    """
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

    # Clean the review title by removing extra spaces
    merged_df["review_title"] = (
        merged_df["review_title"]
        .str.strip()                          # Remove spaces at the beginning and end
        .str.replace(r"\s+", " ", regex=True) # Replace multiple spaces with a single space
    )

    # Clean the review text by removing extra spaces
    merged_df["text"] = (
        merged_df["text"]
        .str.strip()                          # Remove spaces at the beginning and end
        .str.replace(r"\s+", " ", regex=True) # Replace multiple spaces with a single space
    )

    # Combine the review title and review text into one column.
    # When the title is empty, omit the separator so the result is not ". text".
    merged_df["review"] = merged_df.apply(
        lambda row: row["text"] if row["review_title"] == ""
        else row["review_title"] + ". " + row["text"],
        axis=1,
    )

    # Remove extra spaces after combining the text
    merged_df["review"] = (
        merged_df["review"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

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
    train_df, val_df, test_df = split_data(merged_df)

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
