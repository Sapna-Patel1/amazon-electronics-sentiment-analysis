# The file will load the raw datasets, check the columns needed from each, and output some statistics about the dataset
# Final results are a dataframe for the user reviews, a dataframe for the product metadata and the stats for each such as number of rows, missing values, and duplicate rows.

# Import the pandas library to convert the raw data into a dataframe
import pandas as pd
from sklearn.model_selection import train_test_split


def load_metadata(sample_size=10000, chunksize=50000):
    """Load and sample product metadata from the compressed JSONL file.

    Reads the raw metadata in chunks to avoid loading the entire file into
    memory at once, retains only the required columns, and returns a random
    sample of products.

    Args:
        sample_size: Number of products to sample (default 10,000).
        chunksize: Number of rows to read per chunk (default 50,000).

    Returns:
        DataFrame with columns: parent_asin, product_title, main_category,
        average_rating, rating_number.
    """
    # Path to the compressed metadata dataset.
    metadata_path = "data/raw/meta_Electronics.jsonl.gz"

    metadata_chunks = []

    # Read metadata in chunks instead of loading the whole file
    for chunk in pd.read_json(
        metadata_path,
        lines=True,
        compression="gzip",
        chunksize=chunksize,
    ):

        # Keep only required columns
        chunk = chunk[
            [
                "parent_asin",
                "title",
                "main_category",
                "average_rating",
                "rating_number",
            ]
        ]

        metadata_chunks.append(chunk)

    metadata_df = pd.concat(metadata_chunks, ignore_index=True)

    # Rename title column
    metadata_df = metadata_df.rename(
        columns={"title": "product_title"}
    )

    # Randomly sample products
    metadata_df = metadata_df.sample(
        n=sample_size,
        random_state=42
    ).reset_index(drop=True)

    print("\nMetadata Dataset Statistics")
    print("-" * 30)
    print(f"Number of Rows: {len(metadata_df):,}")
    print(f"Number of Columns: {len(metadata_df.columns)}")

    print("\nCheck the datatype of columns")
    print(metadata_df.dtypes)

    print("\nNumber of Missing Values")
    print(metadata_df.isnull().sum())

    print("\nDuplicate Rows")
    duplicates = metadata_df[metadata_df.duplicated(keep=False)]
    print(duplicates.head(10))
    print(f"\nTotal duplicate rows: {len(duplicates)}")

    print("\nMain Categories")
    print(metadata_df["main_category"].value_counts())

    print("\nPreview of the Data")
    print(metadata_df.head(10))

    return metadata_df


def load_reviews(product_ids, sample_size=50000, chunksize=50000):
    """Load and filter reviews for sampled products from the compressed JSONL file.

    Reads the raw reviews in chunks, keeps only reviews belonging to the
    supplied product IDs, removes products with fewer than 10 reviews, and
    returns a random sample up to sample_size.

    Args:
        product_ids: Collection of parent_asin values to keep.
        sample_size: Maximum number of reviews to return (default 50,000).
        chunksize: Number of rows to read per chunk (default 50,000).

    Returns:
        Tuple of (reviews DataFrame, Index of valid product IDs).

    Raises:
        ValueError: If no reviews are found for any of the supplied product IDs.
    """
    # Path to the compressed reviews dataset
    reviews_path = "data/raw/Electronics.jsonl.gz"

    # Store filtered chunks
    review_chunks = []

    # Read the compressed JSONL file in chunks
    for chunk in pd.read_json(
        reviews_path,
        lines=True,
        compression="gzip",
        chunksize=chunksize,
    ):

        # Keep only required columns
        chunk = chunk[
            [
                "parent_asin",
                "title",
                "text",
                "rating",
                "helpful_vote",
                "verified_purchase"
            ]
        ]

        # Rename title column
        chunk = chunk.rename(
            columns={"title": "review_title"}
        )

        # Keep only reviews for sampled products
        filtered_chunk = chunk[
            chunk["parent_asin"].isin(product_ids)
        ]

        if not filtered_chunk.empty:
            review_chunks.append(filtered_chunk)

    if not review_chunks:
        raise ValueError(
            "No reviews found for the sampled product IDs. "
            "Check that product_ids match values in the reviews file."
        )

    # Combine all chunks
    reviews_df = pd.concat(
        review_chunks,
        ignore_index=True
    )

    # Count reviews for each product
    review_counts = reviews_df["parent_asin"].value_counts()

    # Keep only products with more than 10 reviews
    valid_products = review_counts[
        review_counts > 10
    ].index

    # Filter reviews
    reviews_df = reviews_df[
        reviews_df["parent_asin"].isin(valid_products)
    ]

    # Randomly sample reviews
    if len(reviews_df) > sample_size:
        reviews_df = reviews_df.sample(
            n=sample_size,
            random_state=42
        )

    reviews_df = reviews_df.reset_index(drop=True)

    print("\nReview Dataset Statistics")
    print("-" * 30)
    print(f"Number of Rows: {len(reviews_df):,}")
    print(f"Number of Columns: {len(reviews_df.columns)}")

    print("\nCheck the datatype of columns")
    print(reviews_df.dtypes)

    print("\nNumber of Missing Values")
    print(reviews_df.isnull().sum())

    print("\nDuplicate Rows")
    duplicates = reviews_df[reviews_df.duplicated(keep=False)]
    print(duplicates.head(10))
    print(f"\nTotal duplicate rows: {len(duplicates)}")

    print("\nUnique Rating Values")
    print(sorted(reviews_df["rating"].unique()))

    print("\nRating Distribution")
    print(reviews_df["rating"].value_counts().sort_index())

    print("\nPreview of the Data")
    print(reviews_df.head(10))

    return reviews_df, valid_products


def load_processed_data(path="data/processed/preprocessed_reviews.csv"):
    """Load the preprocessed reviews dataset ready for model training.

    Reads the CSV produced by preprocess.py, which already contains the
    sentiment labels (string), integer label (0/1/2), and input_text columns
    required by train.py and evaluate.py.

    Args:
        path: Path to the preprocessed CSV file.

    Returns:
        DataFrame with all preprocessed columns including label and input_text.
    """
    return pd.read_csv(path)


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

    metadata_df = load_metadata()

    product_ids = metadata_df["parent_asin"].unique()

    reviews_df, valid_products = load_reviews(product_ids)

    metadata_df = metadata_df[
        metadata_df["parent_asin"].isin(valid_products)
    ].reset_index(drop=True)

    # Save the sampled datasets as CSVs so that it is easier and faster to run the preprocess.py file
    metadata_df.to_csv("data/raw/metadata_sample.csv", index=False)
    reviews_df.to_csv("data/raw/reviews_sample.csv", index=False)
