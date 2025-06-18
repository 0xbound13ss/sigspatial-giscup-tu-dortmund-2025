import pandas as pd
import numpy as np
from pathlib import Path


def process_mobility_data(input_file, output_file):
    """
    Process mobility data:
    1. Find 10 users with NO data for days 61-75
    2. Find 90 users WITH data for days 61-75
    3. Forward/backward fill days 1-60 for all 100 users
    4. Forward/backward fill days 61-75 ONLY for the 90 users with data
    5. Put 10 users without data first, 90 with data last
    """

    print("Loading data...")
    df = pd.read_csv(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Unique users: {df['uid'].nunique()}")
    print(f"Date range: {df['d'].min()} to {df['d'].max()}")
    print(f"Time slot range: {df['t'].min()} to {df['t'].max()}")

    # Calculate timeline index: (day - 1) * 48 + timeslot
    df["timeline_idx"] = (df["d"] - 1) * 48 + df["t"]

    # Create location string (999,999 becomes NaN)
    df["location"] = df.apply(
        lambda row: (
            f"{int(row['x'])},{int(row['y'])}"
            if row["x"] != 999 and row["y"] != 999
            else np.nan
        ),
        axis=1,
    )

    print("Converting to wide format...")

    # Create wide format using pivot
    wide_data = df.pivot(index="uid", columns="timeline_idx", values="location")

    # Ensure all timeline columns exist (0 to 3599)
    max_timeline = 75 * 48 - 1
    for i in range(max_timeline + 1):
        if i not in wide_data.columns:
            wide_data[i] = np.nan

    # Sort columns and rename
    wide_data = wide_data.reindex(sorted(wide_data.columns), axis=1)
    column_mapping = {i: f"t_{i}" for i in range(max_timeline + 1)}
    wide_data = wide_data.rename(columns=column_mapping)

    # Reset index to make uid a column
    wide_data.reset_index(inplace=True)

    print(f"Wide format shape: {wide_data.shape}")

    # Define time periods
    training_slots = [f"t_{i}" for i in range(0, 60 * 48)]  # Days 1-60: t_0 to t_2879
    prediction_slots = [
        f"t_{i}" for i in range(60 * 48, 75 * 48)
    ]  # Days 61-75: t_2880 to t_3599

    print(f"Training period: {len(training_slots)} slots (days 1-60)")
    print(f"Prediction period: {len(prediction_slots)} slots (days 61-75)")

    # Analyze users based on prediction period data
    print("\nAnalyzing users...")

    users_without_prediction = []
    users_with_prediction = []

    for idx, row in wide_data.iterrows():
        uid = row["uid"]
        prediction_data = row[prediction_slots]
        has_prediction_data = prediction_data.notna().any()

        if has_prediction_data:
            users_with_prediction.append(uid)
        else:
            users_without_prediction.append(uid)

    print(f"Users WITHOUT prediction data: {len(users_without_prediction)}")
    print(f"Users WITH prediction data: {len(users_with_prediction)}")

    # Select exactly 10 users without prediction data
    if len(users_without_prediction) < 10:
        print(
            f"ERROR: Only {len(users_without_prediction)} users without prediction data, need 10!"
        )
        selected_no_prediction = users_without_prediction
    else:
        selected_no_prediction = users_without_prediction[:10]
        print(f"Selected 10 users WITHOUT prediction data: {selected_no_prediction}")

    # Select exactly 90 users with prediction data
    if len(users_with_prediction) < 90:
        print(
            f"ERROR: Only {len(users_with_prediction)} users with prediction data, need 90!"
        )
        selected_with_prediction = users_with_prediction
    else:
        selected_with_prediction = users_with_prediction[:90]
        print(
            f"Selected 90 users WITH prediction data: {selected_with_prediction[:10]}... (showing first 10)"
        )

    # Combine users: NO prediction data first, then WITH prediction data
    final_user_list = selected_no_prediction + selected_with_prediction
    print(f"Total selected users: {len(final_user_list)}")

    # Filter and reorder dataframe
    wide_data_filtered = wide_data[wide_data["uid"].isin(final_user_list)].copy()

    # Create ordering based on our selection
    user_order = {uid: idx for idx, uid in enumerate(final_user_list)}
    wide_data_filtered["sort_order"] = wide_data_filtered["uid"].map(user_order)
    wide_data_filtered = (
        wide_data_filtered.sort_values("sort_order")
        .drop("sort_order", axis=1)
        .reset_index(drop=True)
    )

    print(f"Filtered and reordered shape: {wide_data_filtered.shape}")

    # STEP 1: Forward/backward fill training period (days 1-60) for ALL users
    print("\nStep 1: Filling training period (days 1-60) for ALL users...")

    for idx in range(len(wide_data_filtered)):
        uid = wide_data_filtered.iloc[idx]["uid"]

        # Get training period data for this user
        user_training = wide_data_filtered.iloc[idx][training_slots].copy()

        # Apply forward fill then backward fill
        user_training_filled = user_training.ffill().bfill()

        # Update the dataframe
        wide_data_filtered.iloc[
            idx, wide_data_filtered.columns.get_indexer(training_slots)
        ] = user_training_filled

        if idx < 5:  # Show progress for first 5 users
            missing_before = user_training.isna().sum()
            missing_after = user_training_filled.isna().sum()
            print(
                f"  User {uid}: {missing_before} -> {missing_after} missing training slots"
            )

    # STEP 2: Forward/backward fill prediction period (days 61-75) ONLY for users WITH data
    print(
        "\nStep 2: Filling prediction period (days 61-75) for users WITH data only..."
    )

    users_filled = 0
    for idx in range(len(wide_data_filtered)):
        uid = wide_data_filtered.iloc[idx]["uid"]

        # Only fill prediction period for users who have some prediction data
        if uid in selected_with_prediction:
            user_prediction = wide_data_filtered.iloc[idx][prediction_slots].copy()

            # Apply forward fill then backward fill
            user_prediction_filled = user_prediction.ffill().bfill()

            # Update the dataframe
            wide_data_filtered.iloc[
                idx, wide_data_filtered.columns.get_indexer(prediction_slots)
            ] = user_prediction_filled

            users_filled += 1
            if users_filled <= 5:  # Show progress for first 5
                missing_before = user_prediction.isna().sum()
                missing_after = user_prediction_filled.isna().sum()
                print(
                    f"  User {uid}: {missing_before} -> {missing_after} missing prediction slots"
                )
        else:
            print(f"  User {uid}: SKIPPED (no prediction data - keeping empty)")

    print(f"Filled prediction period for {users_filled} users")

    # Final statistics
    training_filled = wide_data_filtered[training_slots].notna().sum().sum()
    prediction_filled = wide_data_filtered[prediction_slots].notna().sum().sum()

    training_total = len(wide_data_filtered) * len(training_slots)
    prediction_total = len(wide_data_filtered) * len(prediction_slots)

    print(f"\nFinal Statistics:")
    print(f"Users: {len(wide_data_filtered)}")
    print(f"Timeline columns: {max_timeline + 1}")
    print(f"Training period fill rate: {training_filled/training_total*100:.1f}%")
    print(f"Prediction period fill rate: {prediction_filled/prediction_total*100:.1f}%")

    # Show verification of user order
    print(f"\nUser order verification:")
    print(
        f"First 10 users (NO prediction data): {wide_data_filtered['uid'].head(10).tolist()}"
    )
    print(
        f"Last 10 users (WITH prediction data): {wide_data_filtered['uid'].tail(10).tolist()}"
    )

    # Save to CSV
    print(f"\nSaving to {output_file}...")
    wide_data_filtered.to_csv(output_file, index=False)
    print("Processing completed!")

    return wide_data_filtered


if __name__ == "__main__":
    input_file = "city_C_challengedata.csv"
    output_file = "city_C_processed.csv"

    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
    else:
        result_df = process_mobility_data(input_file, output_file)
        print(f"\nFinal shape: {result_df.shape}")
