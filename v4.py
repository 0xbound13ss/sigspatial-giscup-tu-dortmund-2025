import pandas as pd
import numpy as np
from pathlib import Path


def convert_city_c_fixed(input_file, output_file, max_users=100, debug=True):
    """
    Fixed version of the city C converter with proper timeline handling
    """

    print("Loading city C data...")
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique users: {df['uid'].nunique()}")
    print(f"Date range: {df['d'].min()} to {df['d'].max()}")
    print(f"Time slot range: {df['t'].min()} to {df['t'].max()}")

    if debug:
        print(f"\nFirst 10 records:")
        print(df.head(10))

        # Check a specific user's data
        sample_user = df["uid"].iloc[0]
        user_data = df[df["uid"] == sample_user].sort_values(["d", "t"])
        print(f"\nSample user {sample_user} data (first 20 records):")
        print(user_data[["uid", "d", "t", "x", "y"]].head(20))

    # Calculate timeline index correctly
    # Timeline index = (day - 1) * 48 + timeslot
    # Day 1, timeslot 0 = timeline index 0
    # Day 1, timeslot 47 = timeline index 47
    # Day 2, timeslot 0 = timeline index 48
    # etc.
    df["timeline_idx"] = (df["d"] - 1) * 48 + df["t"]

    print(
        f"Timeline index range: {df['timeline_idx'].min()} to {df['timeline_idx'].max()}"
    )
    print(f"Expected timeline range for 75 days: 0 to {75*48-1}")

    # Filter out invalid timeline indices
    valid_timeline = df["timeline_idx"] <= (75 * 48 - 1)
    df = df[valid_timeline]
    print(f"Records with valid timeline: {len(df)}")

    # Focus on training period (days 1-60) for user selection
    training_data = df[(df["d"] <= 60) & (df["x"] != 999) & (df["y"] != 999)].copy()

    print(f"\nTraining period analysis:")
    print(f"Training data (days 1-60, valid coords): {len(training_data)} records")

    # Count valid records per user in training period
    user_counts = training_data["uid"].value_counts()
    print(f"Users with training data: {len(user_counts)}")
    print(
        f"Records per user - min: {user_counts.min()}, max: {user_counts.max()}, avg: {user_counts.mean():.1f}"
    )

    # Select top users with most complete training data
    top_users = user_counts.head(max_users).index.tolist()
    print(f"\nSelected top {len(top_users)} users: {top_users[:10]}...")
    print(
        f"Selected users have {user_counts.head(max_users).min()}-{user_counts.head(max_users).max()} valid training records"
    )

    # Filter dataset to only include selected users
    df_filtered = df[df["uid"].isin(top_users)].copy()
    print(f"Filtered dataset: {len(df_filtered)} records for {len(top_users)} users")

    # Create location string, handling missing data properly
    df_filtered["location"] = df_filtered.apply(
        lambda row: (
            f"{int(row['x'])},{int(row['y'])}"
            if row["x"] != 999 and row["y"] != 999
            else np.nan
        ),
        axis=1,
    )

    # Create wide format DataFrame
    max_timeline = 75 * 48 - 1  # 3599
    timeline_columns = [f"t_{i}" for i in range(max_timeline + 1)]

    print(
        f"Creating wide format: {len(top_users)} users × {len(timeline_columns)} timeline slots"
    )

    # Initialize with all NaN
    wide_data = pd.DataFrame(index=top_users, columns=timeline_columns, dtype="object")

    # Fill in the data efficiently using pivot
    print("Filling in location data...")
    for _, row in df_filtered.iterrows():
        uid = row["uid"]
        timeline_idx = row["timeline_idx"]
        location = row["location"]

        if timeline_idx <= max_timeline and uid in wide_data.index:
            wide_data.at[uid, f"t_{timeline_idx}"] = location

    # Add uid as first column
    wide_data.reset_index(inplace=True)
    wide_data.rename(columns={"index": "uid"}, inplace=True)

    print(f"Wide format shape: {wide_data.shape}")

    # Calculate statistics
    total_cells = len(top_users) * (max_timeline + 1)
    filled_cells = wide_data.iloc[:, 1:].notna().sum().sum()
    fill_rate = filled_cells / total_cells * 100

    # Calculate training vs prediction period statistics
    training_slots = list(range(0, 60 * 48))  # slots 0-2879 (days 1-60)
    prediction_slots = list(range(60 * 48, 75 * 48))  # slots 2880-3599 (days 61-75)

    training_cols = [f"t_{i}" for i in training_slots]
    prediction_cols = [f"t_{i}" for i in prediction_slots]

    training_filled = wide_data[training_cols].notna().sum().sum()
    prediction_filled = wide_data[prediction_cols].notna().sum().sum()

    training_total = len(top_users) * len(training_slots)
    prediction_total = len(top_users) * len(prediction_slots)

    print(f"\nStatistics:")
    print(f"Total users: {len(top_users)}")
    print(f"Timeline slots: {max_timeline + 1}")
    print(f"Overall fill rate: {fill_rate:.2f}%")
    print(
        f"Training period (days 1-60): {training_filled}/{training_total} filled ({training_filled/training_total*100:.1f}%)"
    )
    print(
        f"Prediction period (days 61-75): {prediction_filled}/{prediction_total} filled ({prediction_filled/prediction_total*100:.1f}%)"
    )

    # Show sample around training/prediction boundary
    print(f"\nSample around day 60-61 boundary (timeline slots 2875-2885):")
    boundary_cols = ["uid"] + [f"t_{i}" for i in range(2875, 2885)]
    print(wide_data[boundary_cols].head(3))

    # Show actual early data
    print(f"\nSample of early timeline slots (t_0 to t_10):")
    early_cols = ["uid"] + [f"t_{i}" for i in range(11)]
    sample_data = wide_data[early_cols].head(3)
    print(sample_data)

    # Debug: Check if user 13 has the right data
    if 13 in top_users:
        user_13_wide = wide_data[wide_data["uid"] == 13]
        print(f"\nUser 13 debug - first 20 timeline slots:")
        debug_cols = ["uid"] + [f"t_{i}" for i in range(20)]
        print(user_13_wide[debug_cols])

        # Check user 13's original data
        user_13_orig = df_filtered[df_filtered["uid"] == 13].sort_values(["d", "t"])
        print(f"\nUser 13 original data (first 10 records):")
        print(
            user_13_orig[["uid", "d", "t", "timeline_idx", "x", "y", "location"]].head(
                10
            )
        )

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    wide_data.to_csv(output_file, index=False)
    print("Conversion completed!")

    return wide_data


if __name__ == "__main__":
    input_file = "city_C_challengedata.csv"
    output_file = "city_C_wide_format_fixed.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
    else:
        wide_df = convert_city_c_fixed(
            input_file, output_file, max_users=100, debug=True
        )
