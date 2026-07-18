# The file will load the raw datasets, check the columns needed from each, and output some statistics about the dataset
# Final results are a dataframe for the user reviews, a dataframe for the product metadata and the stats for each such as number of rows, missing values, and duplicate rows.

# Import the pandas library to convert the raw data into a dataframe
import pandas as pd


def load_metadata(sample_size=10000, chunksize=50000):

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

if __name__ == "__main__":

    metadata_df = load_metadata()

    product_ids = metadata_df["parent_asin"].unique()

    reviews_df, valid_products = load_reviews(product_ids)

    metadata_df = metadata_df[
        metadata_df["parent_asin"].isin(valid_products)
    ].reset_index(drop=True)

    # Save the sampled datasets as CSVs so that it is easier and faster to run the preprocess.py file
    metadata_df.to_csv("data/raw/metadata_sample.csv",index=False)
    reviews_df.to_csv("data/raw/reviews_sample.csv",index=False)