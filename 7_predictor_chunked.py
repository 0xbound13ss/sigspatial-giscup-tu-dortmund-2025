import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import settings
from utils import (
    get_xy_list_from_df_simple,
)
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import random
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import os


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

parquet_file = "city_A_challengedata_converted.parquet"
END_TS = settings.TIMESTAMPS_PER_DAY * settings.TRAIN_DAYS - 1

early_stop_stats = {
    "2_under_500": 0,
    "3_under_1000": 0,
    "4_under_1000": 0,
    "timeout": 0,
}

stats_file = "stopping_stats.txt"

if os.path.exists(stats_file):
    with open(stats_file, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":")
                key = key.strip()
                val = val.strip()
                if key in early_stop_stats:
                    early_stop_stats[key] = int(val)
    print(f"Loaded existing early stopping stats from {stats_file}")


def save_prediction_for_user(user_predictions, city_name, team_name="TUDortmund"):
    """Save prediction for a single user to submission file"""
    if len(user_predictions) == 0:
        return

    filename = f"{team_name}_{city_name}_humob25.csv"

    # Check if file exists to determine if we need header
    file_exists = pd.io.common.file_exists(filename)

    # Save in submission format: uid,d,t,x,y
    user_predictions[["uid", "d", "t", "x", "y"]].to_csv(
        filename, mode="a", header=not file_exists, index=False
    )
    print(f"Saved prediction for user {user_predictions['uid'].iloc[0]} to {filename}")


def finalize_submission(city_name, team_name="TUDortmund"):
    """Compress the final submission file"""
    import gzip
    import shutil

    csv_filename = f"{team_name}_{city_name}_humob25.csv"
    gz_filename = f"{team_name}_{city_name}_humob25.csv.gz"

    try:
        with open(csv_filename, "rb") as f_in:
            with gzip.open(gz_filename, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Check submission format
        check_submission_format(gz_filename)

    except Exception as e:
        print(f"Error compressing file: {e}")


def check_submission_format(gz_filename):
    """Basic check of submission format"""
    import gzip

    try:
        with gzip.open(gz_filename, "rt") as f:
            # Read first few lines
            lines = [f.readline().strip() for _ in range(5)]

        print(f"\nSubmission format check for {gz_filename}:")
        print("First 5 lines:")
        for i, line in enumerate(lines):
            if line:
                print(f"  {i}: {line}")

        # Count total lines
        with gzip.open(gz_filename, "rt") as f:
            total_lines = sum(1 for _ in f)

        # Should be header + (3000 users * 720 timestamps) = 2,160,001 lines
        expected_lines = 1 + (
            3000 * 15 * 48
        )  # header + 3000 users * 15 days * 48 timeslots
        print(f"Total lines: {total_lines}")
        print(f"Expected lines: {expected_lines}")

        if total_lines == expected_lines:
            print("✓ Line count looks correct!")
        else:
            print("⚠ Line count may be incorrect")

    except Exception as e:
        print(f"Error checking submission format: {e}")


def get_user_data(uid, parquet_file):
    """Get all data for a specific user using parquet filters"""
    filters = [("uid", "=", uid)]
    table = pq.read_table(parquet_file, filters=filters)
    return table.to_pandas()


def get_user_train_data(uid, parquet_file):
    """Get training data for a user"""
    filters = [("uid", "=", uid), ("ts", "<=", END_TS)]
    table = pq.read_table(parquet_file, filters=filters, columns=["ts", "x", "y"])
    return table.to_pandas()


def get_user_test_data(uid, parquet_file):
    """Get test data for a user"""
    filters = [("uid", "=", uid), ("ts", ">", END_TS)]
    table = pq.read_table(parquet_file, filters=filters, columns=["ts", "x", "y"])
    return table.to_pandas()


def get_unique_users(parquet_file):
    """Get all unique users"""
    table = pq.read_table(parquet_file, columns=["uid"])
    unique_uids = table.column("uid").unique().to_pylist()
    return sorted(unique_uids)


def blend_extension_movements_with_gmm_improved(
    uids_with_scores, parquet_file, n_components=2
):
    """
    Improved blending with similarity weighting and outlier filtering

    Args:
        uids_with_scores: List of (dtw_score, uid) tuples from find_top_similar_users
        parquet_file: Path to parquet file
        n_components: Number of GMM components
    """
    xy_s = []
    weights = []

    print(f"\n=== Processing {len(uids_with_scores)} similar users ===")

    for dtw_score, uid in uids_with_scores:
        user_test_data = get_user_test_data(uid, parquet_file)
        if len(user_test_data) > 0:
            user_test_data = user_test_data.sort_values("ts")
            xy_data = get_xy_list_from_df_simple(user_test_data)

            # Convert to relative movements
            if len(xy_data) > 1:
                first_pos = xy_data[0]
                print(
                    f"User {uid} (DTW={dtw_score:.1f}): first_pos = {first_pos}, trajectory length = {len(xy_data)}"
                )

                relative_moves = [(0, 0)]
                for i in range(1, len(xy_data)):
                    relative_moves.append(
                        (xy_data[i][0] - first_pos[0], xy_data[i][1] - first_pos[1])
                    )

                xy_s.append(relative_moves)

                # Calculate similarity weight (lower DTW = higher weight)
                # Use inverse relationship: better similarity gets higher weight
                weight = 1.0 / (1.0 + dtw_score / 1000.0)  # Normalize DTW scores
                weights.append(weight)

                print(f"  Similarity weight: {weight:.4f}")
                print(f"  Sample relative_moves[0:5] = {relative_moves[:5]}")
                print(f"  Sample relative_moves[-5:] = {relative_moves[-5:]}")

    if not xy_s:
        print("No valid trajectories found")
        return np.array([])

    print(
        f"\nBlending {len(xy_s)} user trajectories with weights: {[f'{w:.3f}' for w in weights]}"
    )

    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    print(f"Normalized weights: {[f'{w:.3f}' for w in weights]}")

    # Blend corresponding positions across users
    max_length = max(len(moves) for moves in xy_s)
    predicted_movements = []

    for pos in range(min(10, max_length)):  # Show first 10 timesteps for debugging
        positions = [xy_s[i][pos] for i in range(len(xy_s)) if pos < len(xy_s[i])]
        position_weights = [weights[i] for i in range(len(xy_s)) if pos < len(xy_s[i])]

        print(f"\nTimestep {pos}:")
        print(f"  Positions to blend: {positions}")
        print(f"  Position weights: {[f'{w:.3f}' for w in position_weights]}")

        if len(positions) > 1:
            positions_array = np.array(positions)
            position_weights_array = np.array(position_weights)

            # Step 1: Remove outliers using IQR method
            filtered_positions, filtered_weights = filter_outliers(
                positions_array, position_weights_array
            )
            print(f"  After outlier filtering: {len(filtered_positions)} positions")

            if len(filtered_positions) == 0:
                # All were outliers, fall back to weighted mean of originals
                result = weighted_mean(positions_array, position_weights_array)
                print(f"  All outliers, using weighted mean: {result}")
                predicted_movements.append(result)
                continue

            # Check for duplicate points
            unique_positions = np.unique(filtered_positions, axis=0)
            print(f"  Unique positions after filtering: {unique_positions.tolist()}")

            if len(unique_positions) == 1:
                # All positions are identical
                result = tuple(filtered_positions[0])
                print(f"  All identical -> result: {result}")
                predicted_movements.append(result)
            elif len(unique_positions) < n_components:
                # Not enough unique positions for GMM, use weighted mean
                result = weighted_mean(filtered_positions, filtered_weights)
                print(f"  Using weighted mean -> result: {result}")
                predicted_movements.append(result)
            else:
                # Use GMM with sample weights
                result = weighted_gmm_sample(
                    filtered_positions, filtered_weights, n_components
                )
                print(f"  Weighted GMM sampled -> result: {result}")
                predicted_movements.append(result)
        elif len(positions) == 1:
            result = positions[0]
            print(f"  Single position -> result: {result}")
            predicted_movements.append(result)
        else:
            print(f"  No positions available, using (0,0)")
            predicted_movements.append((0, 0))

    # Process remaining timesteps without debug output
    for pos in range(10, max_length):
        positions = [xy_s[i][pos] for i in range(len(xy_s)) if pos < len(xy_s[i])]
        position_weights = [weights[i] for i in range(len(xy_s)) if pos < len(xy_s[i])]

        if len(positions) > 1:
            positions_array = np.array(positions)
            position_weights_array = np.array(position_weights)

            # Filter outliers
            filtered_positions, filtered_weights = filter_outliers(
                positions_array, position_weights_array
            )

            if len(filtered_positions) == 0:
                result = weighted_mean(positions_array, position_weights_array)
                predicted_movements.append(result)
                continue

            unique_positions = np.unique(filtered_positions, axis=0)

            if len(unique_positions) == 1:
                predicted_movements.append(tuple(filtered_positions[0]))
            elif len(unique_positions) < n_components:
                result = weighted_mean(filtered_positions, filtered_weights)
                predicted_movements.append(result)
            else:
                result = weighted_gmm_sample(
                    filtered_positions, filtered_weights, n_components
                )
                predicted_movements.append(result)
        elif len(positions) == 1:
            predicted_movements.append(positions[0])
        else:
            predicted_movements.append((0, 0))

    print(f"\nFinal blended movements[0:5]: {predicted_movements[:5]}")
    print(f"Final blended movements[-5:]: {predicted_movements[-5:]}")

    return np.array(predicted_movements)


def filter_outliers(positions, weights, iqr_multiplier=1.5):
    """
    Filter outliers using IQR method

    Args:
        positions: Array of (x, y) positions
        weights: Array of weights for each position
        iqr_multiplier: Multiplier for IQR range (1.5 is standard)

    Returns:
        Filtered positions and weights
    """
    if len(positions) <= 2:
        return positions, weights

    # Calculate distances from centroid
    centroid = np.average(positions, axis=0, weights=weights)
    distances = np.array([np.linalg.norm(pos - centroid) for pos in positions])

    # Calculate IQR
    q25 = np.percentile(distances, 25)
    q75 = np.percentile(distances, 75)
    iqr = q75 - q25

    # Define outlier bounds
    lower_bound = q25 - iqr_multiplier * iqr
    upper_bound = q75 + iqr_multiplier * iqr

    # Filter positions within bounds
    mask = (distances >= lower_bound) & (distances <= upper_bound)

    filtered_positions = positions[mask]
    filtered_weights = weights[mask]

    # Renormalize weights
    if np.sum(filtered_weights) > 0:
        filtered_weights = filtered_weights / np.sum(filtered_weights)

    return filtered_positions, filtered_weights


def weighted_mean(positions, weights):
    """Calculate weighted mean of positions"""
    if len(positions) == 0:
        return (0, 0)

    # Ensure weights sum to 1
    weights = weights / np.sum(weights)

    weighted_pos = np.average(positions, axis=0, weights=weights)
    return tuple(weighted_pos)


def weighted_gmm_sample(positions, weights, n_components):
    """
    Sample from GMM with weighted data points
    """
    try:
        # Create weighted dataset by replicating points based on weights
        # Scale weights to reasonable integers for replication
        scaled_weights = (weights * 100).astype(int)
        scaled_weights = np.maximum(scaled_weights, 1)  # Ensure at least 1 replica

        # Create replicated dataset
        replicated_positions = []
        for pos, weight in zip(positions, scaled_weights):
            replicated_positions.extend([pos] * weight)

        replicated_positions = np.array(replicated_positions)

        # Fit GMM on replicated data
        gmm = GaussianMixture(
            n_components=min(n_components, len(np.unique(positions, axis=0))),
            random_state=42,
        )
        gmm.fit(replicated_positions)

        # Sample from GMM
        sampled_position, _ = gmm.sample(1)
        return tuple(sampled_position[0])

    except Exception as e:
        print(f"  GMM failed ({e}), falling back to weighted mean")
        return weighted_mean(positions, weights)


def apply_movements_to_coordinates(parquet_file, query_uid, predicted_movements):
    user_train_data = get_user_train_data(query_uid, parquet_file)
    if len(user_train_data) == 0:
        return pd.DataFrame()

    # Get last known coordinates
    last_record = user_train_data.loc[user_train_data["ts"].idxmax()]
    last_coord = np.array([last_record["x"], last_record["y"]], dtype=float)

    results = []

    # Since predicted_movements are relative to first position, apply them to last known position
    for i, relative_movement in enumerate(predicted_movements):
        # Apply relative movement to last known position
        new_coord = last_coord + np.array(relative_movement)
        new_coord = np.clip(new_coord, 1, 200)

        ts = END_TS + 1 + i
        day = ts // 48 + 1
        timeslot = ts % 48

        results.append(
            {
                "uid": query_uid,
                "d": day,
                "t": timeslot,
                "x": int(new_coord[0]),
                "y": int(new_coord[1]),
            }
        )

    return pd.DataFrame(results)


def calculate_user_similarity(train_uid, query_user_trajectory):
    """Calculate DTW similarity between query user and a single training user"""
    train_user_data = get_user_train_data(train_uid, parquet_file)

    if len(train_user_data) == 0:
        return (float("inf"), train_uid)

    train_user_data = train_user_data.sort_values("ts")
    train_user_data["day"] = train_user_data["ts"] // 48 + 1

    similarities = []

    # Calculate DTW for each day
    for day in range(1, settings.TRAIN_DAYS + 1):
        query_day_data = [
            pos for pos in query_user_trajectory if pos[0] // 48 + 1 == day
        ]
        train_day_data = train_user_data[train_user_data["day"] == day]

        if len(query_day_data) > 0 and len(train_day_data) > 0:
            q_xy = [(row["x"], row["y"]) for _, row in train_day_data.iterrows()]
            t_xy = [(pos[1], pos[2]) for pos in query_day_data]  # pos = (ts, x, y)

            if len(q_xy) > 0 and len(t_xy) > 0:
                # Use fastdtw instead of yahoo implementation
                dtw_distance, _ = fastdtw(q_xy, t_xy, dist=euclidean)
                similarities.append(dtw_distance)

    # Keep only elements in 25-75 percentile and get mean
    if similarities:
        similarities = np.array(similarities)
        q25, q75 = np.percentile(similarities, [25, 75])
        filtered_similarities = similarities[
            (similarities >= q25) & (similarities <= q75)
        ]
        avg_similarity = (
            np.mean(filtered_similarities)
            if len(filtered_similarities) > 0
            else float("inf")
        )
    else:
        avg_similarity = float("inf")

    return (avg_similarity, train_uid)


def find_top_similar_users(query_uid, candidate_pool_size=5000, start_time=None):
    """Find similar users using new DTW-based selection logic with early stopping"""
    print(f"Finding similar users for {query_uid}...")

    if not start_time:
        start_time = time.time()

    user_start_time = time.time()

    # Get query user's training data
    query_user_data = get_user_train_data(query_uid, parquet_file)
    if len(query_user_data) == 0:
        print(f"No training data found for user {query_uid}")
        return []

    query_user_data = query_user_data.sort_values("ts")
    query_trajectory = [
        (row["ts"], row["x"], row["y"]) for _, row in query_user_data.iterrows()
    ]

    # random sample from 1..settings.TRAIN_USERS
    candidate_users = random.sample(
        range(1, settings.TRAIN_USERS + 1), candidate_pool_size
    )

    print(f"Comparing against {len(candidate_users)} candidate users...")

    similarities = []
    candidates_500 = []
    candidates_1000 = []
    processed_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_uid = {
            executor.submit(
                calculate_user_similarity, train_uid, query_trajectory
            ): train_uid
            for train_uid in candidate_users
        }

        # Process results as they complete
        for future in tqdm(
            as_completed(future_to_uid),
            total=len(candidate_users),
            desc=f"Calculating DTW for user {query_uid} (Elapsed: {time.time() - start_time:.1f}s)",
        ):
            train_uid = future_to_uid[future]
            processed_count += 1

            try:
                dtw_score, uid = future.result()
                if dtw_score != float("inf"):
                    similarities.append((dtw_score, uid))

                    # Check thresholds for early stopping
                    if dtw_score <= 500:
                        candidates_500.append((dtw_score, uid))
                    if dtw_score <= 1000:
                        candidates_1000.append((dtw_score, uid))

                    # Print current top users (sort and show top 10)
                    similarities_sorted = sorted(similarities, key=lambda x: x[0])
                    elapsed = time.time() - start_time
                    print(
                        f"\nProcessed {processed_count}/{len(candidate_users)} - Elapsed: {elapsed:.1f}s - Current top users for {query_uid}:"
                    )
                    for i, (score, user_id) in enumerate(similarities_sorted[:10]):
                        marker = ""
                        if score <= 500:
                            marker = " ⭐"
                        elif score <= 1000:
                            marker = " ✓"
                        print(
                            f"  {i+1:2d}. User {user_id:5d}: DTW = {score:6.1f}{marker}"
                        )

                    print(f"  Found {len(candidates_500)} users with DTW <= 500")
                    print(f"  Found {len(candidates_1000)} users with DTW <= 1000")

                    # Early stopping conditions
                    if len(candidates_500) >= 2:
                        print(
                            f"🎯 Early stopping: Found {len(candidates_500)} users with DTW <= 500"
                        )
                        early_stop_stats["2_under_500"] += 1
                        # Cancel remaining futures
                        for remaining_future in future_to_uid:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    elif len(candidates_1000) >= 3:
                        print(
                            f"🎯 Early stopping: Found {len(candidates_1000)} users with DTW <= 1000"
                        )
                        early_stop_stats["3_under_1000"] += 1
                        # Cancel remaining futures
                        for remaining_future in future_to_uid:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    elif len(candidates_500) + len(candidates_1000) >= 4:
                        print(
                            f"🎯 Early stopping: Found {len(candidates_500) + len(candidates_1000)} users with DTW <= 500 or 1000"
                        )
                        early_stop_stats["4_under_1000"] += 1
                        # Cancel remaining futures
                        for remaining_future in future_to_uid:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break
                    elif time.time() - user_start_time > 15:
                        print(
                            f"⏰ Early stopping: Processing time exceeded 15 seconds ({time.time() - user_start_time:.1f}s for user {train_uid})"
                        )
                        early_stop_stats["timeout"] += 1
                        # Cancel remaining futures
                        for remaining_future in future_to_uid:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

            except Exception as exc:
                print(f"User {train_uid} generated an exception: {exc}")

    with open("early_stopping_stats.txt", "w") as f:
        for key, value in early_stop_stats.items():
            f.write(f"{key}: {value}\n")

    # Sort by DTW score (lower is better)
    similarities.sort(key=lambda x: x[0])

    # Apply final selection logic
    candidates_500 = [sim for sim in similarities if sim[0] <= 500]

    if len(candidates_500) >= 5:
        # Take all users with DTW <= 500
        selected = candidates_500
        print(f"\n✅ Selected {len(selected)} users with DTW <= 500")
    elif len(candidates_500) > 0:
        # Take users with DTW <= 1000
        candidates_1000 = [sim for sim in similarities if sim[0] <= 1000]
        selected = candidates_1000
        print(
            f"\n✅ Selected {len(selected)} users with DTW <= 1000 (only {len(candidates_500)} had <= 500)"
        )
    else:
        # Just take top 5
        selected = similarities[:5]
        print(f"\n✅ Selected top 5 users (no users had DTW <= 500)")

    # Print final selected users with total time
    total_time = time.time() - start_time
    print(f"\nFinal selected users for {query_uid} (Total time: {total_time:.1f}s):")
    for i, (dtw_score, uid) in enumerate(selected):
        print(f"  {i+1}. User {uid}: DTW = {dtw_score:.1f}")

    return selected


def plot_user_with_candidates_and_prediction(
    query_uid,
    top_similar,
    user_predictions,
    parquet_file,
    save_dir="./prediction_plots",
):
    """
    Optimized version with smaller file sizes while maintaining good quality
    """
    print(f"Creating optimized comparison plot for user {query_uid}...")

    candidates = top_similar[:3]
    all_users = [(0, query_uid)] + candidates

    # OPTIMIZATION 1: Better balance of size and quality
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))  # Slightly larger for readability

    # OPTIMIZATION 2: Set style for smaller file sizes
    plt.style.use("default")  # Use default style for efficiency

    for row, (dtw_score, uid) in enumerate(all_users):
        print(f"  Processing user {uid} for plot...")

        if uid == query_uid:
            user_train_data = get_user_train_data(uid, parquet_file)
            user_label = f"Query User {uid}"
            dtw_display = ""  # No DTW for query user

            if len(user_train_data) > 0 and len(user_predictions) > 0:
                pred_data = user_predictions.copy()
                pred_data["ts"] = pred_data["d"] * 48 + pred_data["t"] - 48
                pred_data = pred_data[["ts", "x", "y"]]

                user_data = pd.concat([user_train_data, pred_data], ignore_index=True)
                user_data = user_data.sort_values("ts")
                user_data["day"] = user_data["ts"] // 48 + 1

                train_data = user_data[user_data["day"] <= settings.TRAIN_DAYS]
                pred_data_plot = user_data[user_data["day"] > settings.TRAIN_DAYS]
            else:
                user_data = user_train_data
                train_data = user_train_data
                pred_data_plot = pd.DataFrame()
        else:
            user_data = get_user_data(uid, parquet_file)
            user_label = f"Candidate {uid}"
            dtw_display = f"DTW: {dtw_score:.1f}"  # Show DTW score separately

            if len(user_data) > 0:
                user_data = user_data.sort_values("ts")
                user_data["day"] = user_data["ts"] // 48 + 1
                train_data = user_data[user_data["day"] <= settings.TRAIN_DAYS]
                test_data = user_data[user_data["day"] > settings.TRAIN_DAYS]
            else:
                train_data = pd.DataFrame()
                test_data = pd.DataFrame()

        if len(user_data) == 0:
            continue

        # Calculate deltas
        user_data_with_deltas = user_data.copy()
        if len(user_data_with_deltas) > 1:
            user_data_with_deltas["dx"] = user_data_with_deltas["x"].diff()
            user_data_with_deltas["dy"] = user_data_with_deltas["y"].diff()
            delta_data = user_data_with_deltas.dropna()
        else:
            delta_data = pd.DataFrame()

        # OPTIMIZATION 3: Reduce plot complexity
        ax_traj = axes[row, 0]

        # Plot training data with reduced alpha and smaller markers
        if len(train_data) > 0:
            # OPTIMIZATION 4: Sample data if too many points
            if len(train_data) > 1000:
                train_sample = train_data.sample(n=1000, random_state=42).sort_values(
                    "ts"
                )
            else:
                train_sample = train_data

            ax_traj.plot(
                train_sample["x"],
                train_sample["y"],
                color="blue",
                alpha=0.5,  # Reduced alpha
                linewidth=0.8,  # Thinner lines
                label="Training",
                rasterized=True,  # OPTIMIZATION 5: Rasterize complex plots
            )
            ax_traj.scatter(
                train_data["x"].iloc[0],
                train_data["y"].iloc[0],
                color="green",
                s=50,  # Smaller markers
                marker="o",
                label="Train Start",
                zorder=10,
                edgecolor="black",
                linewidth=1,
            )

        # Handle test/prediction data
        if uid == query_uid:
            if len(pred_data_plot) > 0:
                # OPTIMIZATION 6: Use fewer colors in colormap
                n_colors = min(len(pred_data_plot), 50)  # Limit color variations
                color_indices = np.linspace(
                    0, len(pred_data_plot) - 1, n_colors, dtype=int
                )
                colors = plt.cm.coolwarm(np.linspace(0, 1, n_colors))

                # Simplified line plotting
                ax_traj.plot(
                    pred_data_plot["x"],
                    pred_data_plot["y"],
                    color="red",
                    alpha=0.7,
                    linewidth=1.5,
                    linestyle="--",
                    label="Prediction",
                    rasterized=True,
                )

                # Fewer scatter points
                if len(pred_data_plot) > 20:
                    scatter_indices = np.linspace(
                        0, len(pred_data_plot) - 1, 20, dtype=int
                    )
                    scatter_data = pred_data_plot.iloc[scatter_indices]
                else:
                    scatter_data = pred_data_plot

                ax_traj.scatter(
                    scatter_data["x"],
                    scatter_data["y"],
                    c="red",
                    s=15,  # Smaller markers
                    alpha=0.7,
                    zorder=5,
                    marker="^",
                    rasterized=True,
                )

                # Start/end markers
                ax_traj.scatter(
                    pred_data_plot["x"].iloc[0],
                    pred_data_plot["y"].iloc[0],
                    color="orange",
                    s=60,
                    marker="^",
                    label="Pred Start",
                    zorder=10,
                )
                ax_traj.scatter(
                    pred_data_plot["x"].iloc[-1],
                    pred_data_plot["y"].iloc[-1],
                    color="red",
                    s=60,
                    marker="^",
                    label="Pred End",
                    zorder=10,
                )
        else:
            if len(test_data) > 0:
                # Sample test data if too many points
                if len(test_data) > 500:
                    test_sample = test_data.sample(n=500, random_state=42).sort_values(
                        "ts"
                    )
                else:
                    test_sample = test_data

                ax_traj.plot(
                    test_sample["x"],
                    test_sample["y"],
                    color="red",
                    alpha=0.6,
                    linewidth=1,
                    label="Test",
                    rasterized=True,
                )

                # Start/end markers only
                ax_traj.scatter(
                    test_data["x"].iloc[0],
                    test_data["y"].iloc[0],
                    color="orange",
                    s=60,
                    marker="s",
                    label="Test Start",
                    zorder=10,
                )
                ax_traj.scatter(
                    test_data["x"].iloc[-1],
                    test_data["y"].iloc[-1],
                    color="red",
                    s=60,
                    marker="s",
                    label="Test End",
                    zorder=10,
                )

        # OPTIMIZATION 7: Better axes formatting with DTW scores
        ax_traj.set_xlim(1, 200)
        ax_traj.set_ylim(1, 200)
        ax_traj.set_xlabel("X", fontsize=11)  # Slightly larger fonts
        ax_traj.set_ylabel("Y", fontsize=11)

        # Create clear title with DTW score
        if uid == query_uid:
            title = f"{user_label}"
        else:
            title = f"{user_label}\n{dtw_display}"

        ax_traj.set_title(title, fontsize=11, pad=10)
        ax_traj.grid(True, alpha=0.3, linewidth=0.5)  # Slightly more visible grid
        ax_traj.legend(fontsize=9, markerscale=0.9)

        # Enhanced statistics with DTW score
        if len(user_data) > 0:
            if uid == query_uid:
                stats_text = f"Train: {len(train_data)}\nPred: {len(pred_data_plot)}"
            else:
                stats_text = f"Train: {len(train_data)}\nTest: {len(test_data)}\nDTW: {dtw_score:.1f}"

            ax_traj.text(
                0.02,
                0.98,
                stats_text,
                transform=ax_traj.transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                ),
            )

        # OPTIMIZATION 8: Simplified delta plots
        ax_dx = axes[row, 1]
        if len(delta_data) > 0:
            # Sample delta data if too many points
            if len(delta_data) > 500:
                delta_sample = delta_data.sample(n=500, random_state=42).sort_index()
                time_index = range(len(delta_sample))
                dx_values = delta_sample["dx"]
            else:
                time_index = range(len(delta_data))
                dx_values = delta_data["dx"]

            ax_dx.plot(
                time_index, dx_values, "b-", alpha=0.6, linewidth=0.8, rasterized=True
            )
            ax_dx.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_dx.set_xlabel("Time", fontsize=11)
            ax_dx.set_ylabel("ΔX", fontsize=11)
            ax_dx.set_title("ΔX over Time", fontsize=11)
            ax_dx.grid(True, alpha=0.3, linewidth=0.5)

        # Similar for Delta Y
        ax_dy = axes[row, 2]
        if len(delta_data) > 0:
            if len(delta_data) > 500:
                dy_values = delta_sample["dy"]
            else:
                dy_values = delta_data["dy"]

            ax_dy.plot(
                time_index, dy_values, "g-", alpha=0.6, linewidth=0.8, rasterized=True
            )
            ax_dy.axhline(y=0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_dy.set_xlabel("Time", fontsize=11)
            ax_dy.set_ylabel("ΔY", fontsize=11)
            ax_dy.set_title("ΔY over Time", fontsize=11)
            ax_dy.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout(pad=1.5)  # More padding for readability

    filename = f"{save_dir}/{query_uid}.png"

    # OPTIMIZATION 9: Better balance of quality and file size
    plt.savefig(
        filename,
        dpi=200,  # Better resolution than 150, still reasonable file size
        bbox_inches="tight",
        pad_inches=0.2,  # Bit more padding
        facecolor="white",
        edgecolor="none",
        format="png",
        pil_kwargs={"optimize": True},  # Remove quality param for PNG
    )
    plt.close()
    print(f"  Saved optimized plot: {filename}")


# Update the main function to include plotting and time tracking
def main(start_time):

    # For City A, test users are 147001-150000 (3000 users)
    test_users = list(range(147012, 150001))

    print(f"Processing {len(test_users)} test users for submission")

    # Determine city name from parquet file
    city_name = "cityA"  # Change this based on which city you're processing
    team_name = "TUDortmund"

    successful_predictions = 0
    overall_start_time = time.time()

    for query_uid in tqdm(test_users, desc="Processing test users"):
        user_start_time = time.time()
        print(f"\n=== Predicting for user {query_uid} ===")

        top_similar = find_top_similar_users(query_uid, start_time=start_time)

        if not top_similar:
            print(f"No similar users found for {query_uid}")
            continue

        print("Selected similar users:")
        for dtw_score, uid in top_similar:
            print(f"  User {uid}: DTW = {dtw_score:.1f}")

        print("Blending movements with improved GMM (weighting + outlier filtering)...")

        # Use the improved blending function with scores
        blended_movements = blend_extension_movements_with_gmm_improved(
            top_similar, parquet_file, n_components=2  # Pass tuples with scores
        )

        if len(blended_movements) > 0:
            print(f"Generated {len(blended_movements)} blended movements")

            # Convert to coordinates
            user_predictions = apply_movements_to_coordinates(
                parquet_file, query_uid, blended_movements
            )

            if len(user_predictions) > 0:
                # Create comparison plot with predictions
                plot_user_with_candidates_and_prediction(
                    query_uid, top_similar, user_predictions, parquet_file
                )
                # Save immediately for this user
                save_prediction_for_user(user_predictions, city_name, team_name)
                successful_predictions += 1

        user_elapsed = time.time() - user_start_time
        overall_elapsed = time.time() - overall_start_time
        print(
            f"✅ User {query_uid} completed in {user_elapsed:.1f}s (Total elapsed: {overall_elapsed:.1f}s)"
        )

    total_time = time.time() - overall_start_time
    print(
        f"\nGenerated predictions for {successful_predictions}/{len(test_users)} users"
    )
    print(f"Total processing time: {total_time:.1f} seconds")

    # Finalize submission file
    if successful_predictions > 0:
        finalize_submission(city_name, team_name)
    else:
        print("No predictions generated - no submission file created")


if __name__ == "__main__":
    start_time = time.time()
    main(start_time)
    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")
