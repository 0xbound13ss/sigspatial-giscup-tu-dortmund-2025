import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from config import settings
from geobleu_optimized import geobleu_by_day


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


def find_most_similar_user(query_uid, base_uids, base_movements_df):
    """Find most similar user using GEO-BLEU on relative movements"""
    query_movements = base_movements_df[(base_movements_df["uid"] == query_uid)]
    best_similarity = -1
    best_uid = None
    for base_uid in tqdm(base_uids, desc=f"Finding similar user for {query_uid}"):
        base_movements = base_movements_df[(base_movements_df["uid"] == base_uid)]
        similarity = geobleu_by_day(
            pred_traj=query_movements[["day", "time", "dx", "dy"]].values.tolist(),
            ref_traj=base_movements[["day", "time", "dx", "dy"]].values.tolist(),
        )
        if similarity > best_similarity:
            best_similarity = similarity
            best_uid = base_uid
    return best_uid, best_similarity


# def get_extension_movements(best_uid, extension_movements_df):
#     """Get extension movements from most similar user"""

#     # Get movements from ts=2800 onwards (prediction period)
#     extension_movements = extension_movements_df[
#         (extension_movements_df["uid"] == best_uid)
#         & (extension_movements_df["ts"] > 2799)
#     ].sort_values("ts")

#     if len(extension_movements) == 0:
#         return np.array([])

#     return extension_movements[["dx", "dy"]].values


# def apply_movements_to_coordinates(
#     df, query_uid, predicted_movements, train_end_ts=2799
# ):
#     """Apply predicted movements to get final coordinates"""

#     # Get last training coordinate
#     last_coord = df[(df["uid"] == query_uid) & (df["ts"] == train_end_ts)][
#         ["x", "y"]
#     ].values

#     if len(last_coord) == 0:
#         print(f"No training data found for user {query_uid} at ts={train_end_ts}")
#         return pd.DataFrame()

#     last_coord = last_coord[0].astype(float)

#     # Apply movements
#     results = []
#     current_coord = last_coord.copy()

#     for i, movement in enumerate(predicted_movements):
#         current_coord += movement
#         ts = train_end_ts + 1 + i

#         results.append(
#             {"uid": query_uid, "ts": ts, "x": current_coord[0], "y": current_coord[1]}
#         )

#     return pd.DataFrame(results)


def main():
    # Load data
    print("Loading dataset...")
    df = pd.read_parquet("city_B_challengedata_converted.parquet")
    print(f"Loaded {len(df):,} rows")

    # Define users
    query_uids = list(range(27001, 27011))  # First 10 users
    base_uids = list(range(1, 27001))  # All base users

    # Compute relative movements as DataFrame
    movements_df = compute_relative_movements_df(df)
    print(f"Computed {len(movements_df):,} movement vectors")
    print(
        f"Movement ts range: {movements_df['ts'].min()} to {movements_df['ts'].max()}"
    )

    base_movements_df = movements_df[
        (movements_df["ts"] >= 1) & (movements_df["ts"] <= END_TS)
    ]
    base_trajectories = precompute_base_trajectories(base_movements_df)
    print(f"Pre-computed trajectories for {len(base_trajectories)} base users")

    for query_uid in query_uids:
        print(f"\n=== Predicting for user {query_uid} ===")

        # Find most similar user (top-1)
        best_uid, similarity = find_most_similar_user(
            query_uid, base_uids, base_movements_df
        )

        print(f"Most similar user: {best_uid} (GEO-BLEU: {similarity:.6f})")

        # predicted_movements = get_extension_movements(best_uid, extension_movements_df)

        # print(
        #     f"Using {len(predicted_movements)} extension movements from user {best_uid}"
        # )

        # Convert to coordinates
        # user_predictions = apply_movements_to_coordinates(
        #     df, query_uid, predicted_movements
        # )


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
