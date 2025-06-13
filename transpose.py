import pandas as pd
import numpy as np
from pathlib import Path


def convert_city_c_to_wide_format(input_file, output_file):
    """
    Convert city C mobility data from long format to wide format.

    Args:
        input_file (str): Path to input CSV file (city_C_challengedata.csv)
        output_file (str): Path to output CSV file
    """

    print("Loading city C data...")
    # Load the data
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Unique users: {df['uid'].nunique()}")
    print(f"Date range: {df['d'].min()} to {df['d'].max()}")
    print(f"Time slot range: {df['t'].min()} to {df['t'].max()}")

    # Create timeline column index
    # Formula: timeline_index = (day - 1) * 48 + timeslot
    df["timeline_idx"] = (df["d"] - 1) * 48 + df["t"]

    print(
        f"Timeline index range: {df['timeline_idx'].min()} to {df['timeline_idx'].max()}"
    )

    # Create location string by combining x,y coordinates
    # Handle missing data (999,999) as NaN
    df["location"] = df.apply(
        lambda row: (
            f"{int(row['x'])},{int(row['y'])}"
            if row["x"] != 999 and row["y"] != 999
            else np.nan
        ),
        axis=1,
    )

    # Get all users and all possible timeline indices
    all_users = sorted(df["uid"].unique())
    max_timeline = 75 * 48 - 1  # 0-indexed, so 3599 is the last column

    print(
        f"Creating wide format for {len(all_users)} users and {max_timeline + 1} timeline slots..."
    )

    # Create the wide format dataframe
    # Initialize with NaN values
    wide_data = pd.DataFrame(
        index=all_users,
        columns=[f"t_{i}" for i in range(max_timeline + 1)],
        dtype="object",
    )

    # Fill in the data
    for _, row in df.iterrows():
        uid = row["uid"]
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

    # Show sample of the wide format
    print(f"\nSample of wide format (first 5 users, first 10 timeline slots):")
    print(wide_data.iloc[:5, :11])  # uid + first 10 timeline columns

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
        wide_df = convert_city_c_to_wide_format(input_file, output_file)

        print(f"\nConversion completed! Output saved as '{output_file}'")
