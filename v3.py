import pandas as pd
import numpy as np
from pathlib import Path


def convert_city_c_to_wide_format(input_file, output_file, max_users=1000):
    """
    Convert city C mobility data from long format to wide format.
    Select only users with the most complete data in the first 60 days (training period).

    Args:
        input_file (str): Path to input CSV file (city_C_challengedata.csv)
        output_file (str): Path to output CSV file
        max_users (int): Maximum number of users to select (default: 1000)
    """

    print("Loading city C data...")
    # Load the data
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique users: {df['uid'].nunique()}")
    print(f"Date range: {df['d'].min()} to {df['d'].max()}")
    print(f"Time slot range: {df['t'].min()} to {df['t'].max()}")

    # Focus on training period (days 1-60) and filter out missing data (999,999)
    training_data = df[(df["d"] <= 60) & (df["x"] != 999) & (df["y"] != 999)].copy()
    prediction_period = df[df["d"] > 60].copy()  # Days 61-75 for context

    print(f"\nData split:")
    print(f"Training period (days 1-60): {len(training_data)} valid records")
    print(f"Prediction period (days 61-75): {len(prediction_period)} records")

    # Count valid records per user in training period
    user_counts = training_data["uid"].value_counts()
    print(f"\nUser data completeness in training period (days 1-60):")
    print(f"Users with valid training data: {len(user_counts)}")
    print(f"Max records per user: {user_counts.max()}")
    print(f"Min records per user: {user_counts.min()}")
    print(f"Avg records per user: {user_counts.mean():.1f}")
    print(f"Expected max records (60 days * 48 slots): {60 * 48}")

    # Select top users with most complete training data
    top_users = user_counts.head(max_users).index.tolist()
    print(f"\nSelected top {len(top_users)} users with most complete training data")
    print(
        f"Selected users have {user_counts.head(max_users).min()}-{user_counts.head(max_users).max()} valid training records"
    )

    # Calculate fill rate for selected users in training period
    max_possible_training = 60 * 48  # 2880 possible slots
    avg_fill_rate = (user_counts.head(max_users).mean() / max_possible_training) * 100
    print(
        f"Average fill rate for selected users in training period: {avg_fill_rate:.1f}%"
    )

    # Filter dataset to only include selected users
    df_filtered = df[df["uid"].isin(top_users)].copy()
    print(f"Filtered dataset: {len(df_filtered)} records for {len(top_users)} users")

    # Create timeline column index
    # Formula: timeline_index = (day - 1) * 48 + timeslot
    df_filtered["timeline_idx"] = (df_filtered["d"] - 1) * 48 + df_filtered["t"]

    print(
        f"Timeline index range: {df_filtered['timeline_idx'].min()} to {df_filtered['timeline_idx'].max()}"
    )

    # Create location string by combining x,y coordinates
    # Handle missing data (999,999) as NaN
    df_filtered["location"] = df_filtered.apply(
        lambda row: (
            f"{int(row['x'])},{int(row['y'])}"
            if row["x"] != 999 and row["y"] != 999
            else np.nan
        ),
        axis=1,
    )

    # Get selected users and all possible timeline indices
    all_users = sorted(top_users)
    max_timeline = 75 * 48 - 1  # 0-indexed, so 3599 is the last column

    print(
        f"Creating wide format for {len(all_users)} users and {max_timeline + 1} timeline slots..."
    )

    # Create the wide format dataframe more efficiently
    # Initialize with NaN values
    wide_data = pd.DataFrame(
        index=all_users,
        columns=[f"t_{i}" for i in range(max_timeline + 1)],
        dtype="object",
    )

    # Fill in the data more efficiently using groupby
    print("Filling in location data...")
    for uid in all_users:
        user_data = df_filtered[df_filtered["uid"] == uid]
        for _, row in user_data.iterrows():
            timeline_idx = row["timeline_idx"]
            location = row["location"]

            if timeline_idx <= max_timeline:  # Safety check
                wide_data.at[uid, f"t_{timeline_idx}"] = location

    # Reset index to make uid a column
    wide_data.reset_index(inplace=True)
    wide_data.rename(columns={"index": "uid"}, inplace=True)

    print(f"Wide format shape: {wide_data.shape}")
    print(f"Saving to {output_file}...")

    # Save to CSV
    wide_data.to_csv(output_file, index=False)

    print("Conversion completed!")

    # Print some statistics
    total_cells = len(all_users) * (max_timeline + 1)
    filled_cells = wide_data.iloc[:, 1:].notna().sum().sum()  # Exclude uid column
    fill_rate = filled_cells / total_cells * 100

    print(f"\nStatistics:")
    print(f"Total users: {len(all_users)}")
    print(f"Timeline slots: {max_timeline + 1}")
    print(f"Total cells: {total_cells}")
    print(f"Filled cells: {filled_cells}")
    print(f"Fill rate: {fill_rate:.2f}%")

    # Show data distribution across periods
    training_slots = list(range(0, 60 * 48))  # slots 0-2879
    prediction_slots = list(range(60 * 48, 75 * 48))  # slots 2880-3599

    training_fill = wide_data[[f"t_{i}" for i in training_slots]].notna().sum().sum()
    prediction_fill = (
        wide_data[[f"t_{i}" for i in prediction_slots]].notna().sum().sum()
    )

    training_total = len(all_users) * len(training_slots)
    prediction_total = len(all_users) * len(prediction_slots)

    print(f"\nData distribution:")
    print(
        f"Training period (days 1-60): {training_fill}/{training_total} filled ({training_fill/training_total*100:.1f}%)"
    )
    print(
        f"Prediction period (days 61-75): {prediction_fill}/{prediction_total} filled ({prediction_fill/prediction_total*100:.1f}%)"
    )

    # Show sample focusing on training and prediction boundary
    print(f"\nSample around day 60-61 boundary (timeline slots 2875-2885):")
    boundary_cols = ["uid"] + [f"t_{i}" for i in range(2875, 2885)]
    print(wide_data[boundary_cols].head())

    return wide_data


def analyze_timeline_distribution(df):
    """
    Analyze the distribution of data across timeline slots.
    """
    print("\nAnalyzing timeline distribution...")

    # Create timeline index
    df["timeline_idx"] = (df["d"] - 1) * 48 + df["t"]

    # Count records per timeline slot
    timeline_counts = df["timeline_idx"].value_counts().sort_index()

    print(f"Timeline slots with data: {len(timeline_counts)}")
    print(f"Timeline slots with most data: {timeline_counts.head()}")
    print(f"Timeline slots with least data: {timeline_counts.tail()}")

    # Analyze missing patterns
    missing_slots = set(range(75 * 48)) - set(timeline_counts.index)
    print(f"Timeline slots with no data: {len(missing_slots)}")

    return timeline_counts


if __name__ == "__main__":
    # File paths - adjust these as needed
    input_file = "city_C_challengedata.csv"
    output_file = "city_C_wide_format.csv"

    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        print(
            "Please make sure the file is in the current directory or adjust the path."
        )
    else:
        # Load and analyze the data first
        df = pd.read_csv(input_file)
        analyze_timeline_distribution(df)

        # Convert to wide format
        wide_df = convert_city_c_to_wide_format(input_file, output_file, max_users=1000)

        print(f"\nConversion completed! Output saved as '{output_file}'")
