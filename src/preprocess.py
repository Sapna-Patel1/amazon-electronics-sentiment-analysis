# The file preprocesses the Amazon Electronics dataset by merging the reviews and metadata datasets, cleaning the data, creating sentiment labels, and splitting the data into training, validation, and testing datasets.

# Import the pandas library for data manipulation
import pandas as pd
# Splitting the data into test and validation sets
from sklearn.model_selection import train_test_split

# Create a function to assign sentiment labels
def get_sentiment(rating):

    # Ratings below 3 are negative
    if rating < 3:
        return "Negative"

    # A rating of 3 is neutral
    elif rating == 3:
        return "Neutral"

    # Ratings above 3 are positive
    else:
        return "Positive"

# Split the dataset into training, validation, and testing sets
def split_data(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Calculate the percentage of data for the testing set
    test_size = 1.0 - train_size - val_size

    # Split the data into training and temporary datasets
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_size + test_size),
        random_state=seed,
        stratify=df["sentiment"]
    )

    # Calculate the validation size relative to the temporary dataset
    relative_val = val_size / (val_size + test_size)

    # Split the temporary dataset into validation and testing datasets
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - relative_val),
        random_state=seed,
        stratify=temp_df["sentiment"]
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )

def preprocess_data():

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

    # Combine the review title and review text into one column
    merged_df["review"] = (
        merged_df["review_title"] + ". " + merged_df["text"]
    )

    # Remove extra spaces after combining the text
    merged_df["review"] = (
        merged_df["review"]
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Create the sentiment column
    merged_df["sentiment"] = merged_df["rating"].apply(get_sentiment)

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