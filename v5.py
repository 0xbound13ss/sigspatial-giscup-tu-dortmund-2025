import pandas as pd
import numpy as np
from pathlib import Path


def convert_top_users_by_density(input_file, output_file, top_n=1000):
    """
    Convert city C data to wide format, selecting top N users by total data points
    """

    print("Loading city C data...")
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Unique users: {df['uid'].nunique()}")

    # Count total records per user (including missing data)
    user_record_counts = df["uid"].value_counts().sort_values(ascending=False)

    print(f"\nUser record count statistics:")
    print(f"Max records per user: {user_record_counts.max()}")
    print(f"Min records per user: {user_record_counts.min()}")
    print(f"Avg records per user: {user_record_counts.mean():.1f}")
    print(f"Median records per user: {user_record_counts.median():.1f}")

    # Select top N users with most data points
    top_users = user_record_counts.head(top_n).index.tolist()

    print(f"\nSelected top {len(top_users)} users by data density:")
    print(f"Top user has {user_record_counts.iloc[0]} records")
    print(f"#{top_n} user has {user_record_counts.iloc[top_n-1]} records")
    print(f"Selected users: {top_users[:10]}... (showing first 10)")

    # Filter to selected users only
    df_filtered = df[df["uid"].isin(top_users)].copy()
    print(f"Filtered dataset: {len(df_filtered)} records")

    # Calculate timeline index
    df_filtered["timeline_idx"] = (df_filtered["d"] - 1) * 48 + df_filtered["t"]

    # Create location string (handle missing data as NaN)
    df_filtered["location"] = df_filtered.apply(
        lambda row: (
            f"{int(row['x'])},{int(row['y'])}"
            if row["x"] != 999 and row["y"] != 999
            else np.nan
        ),
        axis=1,
    )

    # Create wide format - 75 days * 48 timeslots = 3600 columns
    max_timeline = 75 * 48 - 1
    timeline_columns = [f"t_{i}" for i in range(max_timeline + 1)]

    print(
        f"Creating wide format: {len(top_users)} users × {len(timeline_columns)} timeline columns"
    )

    # Use pivot_table for efficient conversion
    # Create a temporary DataFrame for pivoting
    pivot_data = df_filtered[["uid", "timeline_idx", "location"]].copy()

    # Remove duplicates (in case same user has multiple records for same timeline slot)
    pivot_data = pivot_data.drop_duplicates(
        subset=["uid", "timeline_idx"], keep="first"
    )

    # Create the wide format using pivot
    wide_data = pivot_data.pivot(index="uid", columns="timeline_idx", values="location")

    # Ensure all timeline columns exist (fill missing ones with NaN)
    for i in range(max_timeline + 1):
        if i not in wide_data.columns:
            wide_data[i] = np.nan

    # Sort columns by timeline index
    wide_data = wide_data.reindex(sorted(wide_data.columns), axis=1)

    # Rename columns to t_0, t_1, etc.
    column_mapping = {i: f"t_{i}" for i in range(max_timeline + 1)}
    wide_data = wide_data.rename(columns=column_mapping)

    # Reset index to make uid a column
    wide_data.reset_index(inplace=True)

    # Ensure we have exactly the users we want (in case pivot missed some)
    wide_data = wide_data[wide_data["uid"].isin(top_users)]

    # Sort by data density (descending) to match the original ranking
    user_order = {uid: idx for idx, uid in enumerate(top_users)}
    wide_data["sort_order"] = wide_data["uid"].map(user_order)
    wide_data = wide_data.sort_values("sort_order").drop("sort_order", axis=1)

    print(f"Final wide format shape: {wide_data.shape}")

    # Calculate statistics
    total_cells = len(wide_data) * (max_timeline + 1)
    filled_cells = wide_data.iloc[:, 1:].notna().sum().sum()  # Exclude uid column
    fill_rate = filled_cells / total_cells * 100

    # Training vs prediction period stats
    training_slots = list(range(0, 60 * 48))  # Days 1-60
    prediction_slots = list(range(60 * 48, 75 * 48))  # Days 61-75

    training_cols = [f"t_{i}" for i in training_slots]
    prediction_cols = [f"t_{i}" for i in prediction_slots]

    training_filled = wide_data[training_cols].notna().sum().sum()
    prediction_filled = wide_data[prediction_cols].notna().sum().sum()

    training_total = len(wide_data) * len(training_slots)
    prediction_total = len(wide_data) * len(prediction_slots)

    print(f"\nStatistics:")
    print(f"Users: {len(wide_data)}")
    print(f"Timeline columns: {max_timeline + 1}")
    print(f"Overall fill rate: {fill_rate:.2f}%")
    print(f"Training period fill rate: {training_filled/training_total*100:.1f}%")
    print(f"Prediction period fill rate: {prediction_filled/prediction_total*100:.1f}%")

    # Show sample of data
    print(f"\nSample data (first 3 users, first 15 timeline slots):")
    sample_cols = ["uid"] + [f"t_{i}" for i in range(15)]
    print(wide_data[sample_cols].head(3))

    # Check a specific user to verify correctness
    if len(wide_data) > 0:
        sample_uid = wide_data.iloc[0]["uid"]
        print(f"\nVerification for user {sample_uid}:")

        # Get original data for this user
        orig_user_data = df_filtered[df_filtered["uid"] == sample_uid].sort_values(
            ["d", "t"]
        )
        print(f"Original records: {len(orig_user_data)}")

        # Show first few original records
        print("First 5 original records:")
        print(orig_user_data[["d", "t", "timeline_idx", "x", "y", "location"]].head())

        # Show corresponding wide format data
        user_wide = wide_data[wide_data["uid"] == sample_uid]
        first_filled_cols = []
        for i in range(50):  # Check first 50 timeline slots
            col_name = f"t_{i}"
            if pd.notna(user_wide[col_name].iloc[0]):
                first_filled_cols.append(col_name)
                if len(first_filled_cols) >= 5:
                    break

        if first_filled_cols:
            verify_cols = ["uid"] + first_filled_cols
            print(f"First filled timeline slots in wide format:")
            print(user_wide[verify_cols])

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    wide_data.to_csv(output_file, index=False)
    print("Conversion completed successfully!")

    return wide_data


if __name__ == "__main__":
    input_file = "city_C_challengedata.csv"
    output_file = "city_C_top_users_wide.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
    else:
        wide_df = convert_top_users_by_density(input_file, output_file, top_n=1000)
