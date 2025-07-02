import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from config import settings
from geobleu_optimized import geobleu_by_day
from sklearn.mixture import GaussianMixture


END_TS = settings.TIMESTAMPS_PER_DAY * settings.TRAIN_DAYS - 1


def compute_relative_movements_df(df):
    """Convert absolute coordinates to relative movement vectors, return as DataFrame"""
    relative_start_time = time.time()
    print("Computing relative movements...")
    df_sorted = df.sort_values(["uid", "ts"])
    df_sorted[["dx", "dy"]] = df_sorted.groupby("uid")[["x", "y"]].diff()
    movements_df = df_sorted.dropna(subset=["dx", "dy"]).copy()
    movements_df = movements_df[["uid", "ts", "dx", "dy"]].reset_index(drop=True)
    movements_df["day"] = movements_df["ts"] // 48 + 1
    movements_df["time"] = movements_df["ts"] % 48
    print(
        f"Computed relative movements in {time.time() - relative_start_time:.2f} seconds"
    )
    return movements_df


def precompute_base_trajectories(base_movements_df):
    """Pre-compute trajectories for all base users for efficiency"""
    print("Pre-computing base user trajectories...")
    precompute_start = time.time()

    # Filter to only base users at once
    base_users = base_movements_df["uid"].max() - settings.TEST_USERS
    base_only_df = base_movements_df[base_movements_df["uid"] < base_users]

    # Group by uid and convert to dict of lists in one operation
    base_trajectories = {}
    for uid, group in tqdm(base_only_df.groupby("uid"), desc="Processing base users"):
        base_trajectories[uid] = group[["day", "time", "dx", "dy"]].values.tolist()

    print(f"Pre-computed trajectories in {time.time() - precompute_start:.2f} seconds")
    return base_trajectories


def find_top_similar_users(query_uid, base_trajectories, base_movements_df, top_k=5):
    """Find top K most similar users using GEO-BLEU on relative movements"""
    query_movements = base_movements_df[(base_movements_df["uid"] == query_uid)]
    query_trajectory = query_movements[["day", "time", "dx", "dy"]].values.tolist()
    similarities = []

    base_users = base_movements_df["uid"].max() - settings.TEST_USERS

    for base_uid in tqdm(
        list(range(1, base_users + 1)),
        desc=f"Finding similar users for {query_uid}",
    ):
        base_trajectory = base_trajectories[base_uid]
        similarity = geobleu_by_day(
            pred_traj=query_trajectory,
            ref_traj=base_trajectory,
        )
        similarities.append((base_uid, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def get_extension_movements(uid, extension_movements_df):
    """Get extension movements from a user for prediction period"""
    extension_movements = extension_movements_df[
        (extension_movements_df["uid"] == uid)
    ].sort_values("ts")

    if len(extension_movements) == 0:
        return np.array([])

    return extension_movements[["dx", "dy"]].values


def blend_movements_with_gmm(movement_sets, n_components=2):
    """
    Blend multiple movement sets using Gaussian Mixture Model

    Args:
        movement_sets: List of movement arrays from different users
        n_components: Number of Gaussian components for GMM

    Returns:
        Sampled movements from the fitted GMM
    """
    if not movement_sets or all(len(moves) == 0 for moves in movement_sets):
        return np.array([])

    # Combine all movements
    all_movements = np.vstack([moves for moves in movement_sets if len(moves) > 0])

    if len(all_movements) == 0:
        return np.array([])

    # Fit GMM
    n_components = min(n_components, len(all_movements))
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=42
    )
    gmm.fit(all_movements)

    max_length = max(len(moves) for moves in movement_sets)

    # Sample from GMM
    sampled_movements, _ = gmm.sample(max_length)

    return sampled_movements


def apply_movements_to_coordinates(
    df, query_uid, predicted_movements, train_end_ts=END_TS
):
    """Apply predicted movements to get final coordinates"""
    # Get last training coordinate
    last_coord = df[(df["uid"] == query_uid) & (df["ts"] == train_end_ts)][
        ["x", "y"]
    ].values

    last_coord = last_coord[0].astype(float)

    results = []
    current_coord = last_coord.copy()

    for i, movement in enumerate(predicted_movements):
        current_coord += movement

        # Clip coordinates to valid range [1, 200]
        current_coord = np.clip(current_coord, 1, 200)

        ts = train_end_ts + 1 + i
        day = ts // 48 + 1
        time_slot = ts % 48

        results.append(
            {
                "uid": query_uid,
                "ts": ts,
                "day": day,
                "time": time_slot,
                "x": current_coord[0],
                "y": current_coord[1],
            }
        )

    return pd.DataFrame(results)


def main():
    # Load data
    print("Loading dataset...")
    df = pd.read_parquet("city_B_challengedata_converted.parquet")
    print(f"Loaded {len(df):,} rows")

    # Define users
    query_uids = list(range(27001, 27011))  # First 10 users

    # Compute relative movements as DataFrame
    movements_df = compute_relative_movements_df(df)
    print(f"Computed {len(movements_df):,} movement vectors")
    print(
        f"Movement ts range: {movements_df['ts'].min()} to {movements_df['ts'].max()}"
    )

    base_movements_df = movements_df[
        (movements_df["ts"] >= 1) & (movements_df["ts"] <= END_TS)
    ]
    extension_movements_df = movements_df[movements_df["ts"] > END_TS]

    base_trajectories = precompute_base_trajectories(base_movements_df)
    print(f"Pre-computed trajectories for {len(base_trajectories)} base users")

    all_predictions = []

    for query_uid in query_uids:
        print(f"\n=== Predicting for user {query_uid} ===")

        top_similar = find_top_similar_users(
            query_uid, base_trajectories, base_movements_df, top_k=5
        )

        print("Top 5 similar users:")
        for uid, similarity in top_similar:
            print(f"  User {uid}: GEO-BLEU = {similarity:.6f}")

        # Get extension movements from top 5 users
        movement_sets = []
        for uid, similarity in top_similar:
            movements = get_extension_movements(uid, extension_movements_df)
            if len(movements) > 0:
                movement_sets.append(movements)
                print(f"User {uid}: {len(movements)} extension movements")

        if not movement_sets:
            print(f"No extension movements found for user {query_uid}")
            continue

        print("Blending movements with GMM...")
        blended_movements = blend_movements_with_gmm(movement_sets, n_components=2)
        print(f"Generated {len(blended_movements)} blended movements")

        # Convert to coordinates
        user_predictions = apply_movements_to_coordinates(
            df, query_uid, blended_movements
        )

        print(f"Generated {len(user_predictions)} coordinate predictions")
        print(
            f"Coordinate range: X=[{user_predictions['x'].min():.1f}, {user_predictions['x'].max():.1f}], "
            f"Y=[{user_predictions['y'].min():.1f}, {user_predictions['y'].max():.1f}]"
        )
        all_predictions.append(user_predictions)

    # Combine all predictions
    if all_predictions:
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        print(f"\n=== Final Results ===")
        print(f"Total predictions: {len(final_predictions)}")
        print(f"Users predicted: {final_predictions['uid'].nunique()}")

        # Save predictions
        output_file = "gmm_predictions.csv"
        final_predictions[["uid", "day", "time", "x", "y"]].to_csv(
            output_file, index=False, float_format="%.0f"
        )
        print(f"Predictions saved to {output_file}")
    else:
        print("No predictions generated for any user")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal time: {time.time() - start_time:.1f} seconds")

# (sigspatial-giscup-tu-dortmund-2025-py3.13) fortisai@mbp:~/projects/sigspatial-giscup-tu-dortmund-2025$ python 3_predictor.py
# Loading dataset...
# Loaded 108,000,000 rows
# Computing relative movements...
# Computed relative movements in 23.16 seconds
# Computed 107,970,000 movement vectors
# Movement ts range: 1 to 3599
# Pre-computing base user trajectories...
# Processing base users: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 26999/26999 [01:39<00:00, 271.39it/s]
# Pre-computed trajectories in 103.73 seconds
# Pre-computed trajectories for 26999 base users

# === Predicting for user 27001 ===
# Finding similar user for 27001:   1%|▋                                                                                      | 220/27000 [00:37<1:14:22,  6.00it/s]
