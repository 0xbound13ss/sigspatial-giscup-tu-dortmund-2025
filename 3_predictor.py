import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from config import settings
from utils import (
    get_algo,
    get_xy_list_from_df_simple,
)
from sklearn.mixture import GaussianMixture
from concurrent.futures import ThreadPoolExecutor, as_completed


df = None


END_TS = settings.TIMESTAMPS_PER_DAY * settings.TRAIN_DAYS - 1


def blend_extension_movements_with_gmm(uids, n_components=2):
    xy_s = [
        get_xy_list_from_df_simple(
            df[df["uid"] == uid][df["day"] > settings.TRAIN_DAYS]
        )
        for uid in uids
    ]

    for i, _ in enumerate(xy_s):
        first_pos = xy_s[i][0]
        for j in range(1, len(xy_s[i])):
            xy_s[i][j] = (
                xy_s[i][j][0] - first_pos[0],
                xy_s[i][j][1] - first_pos[1],
            )
        xy_s[i][0] = (0, 0)

    # Blend corresponding positions across users
    max_length = max(len(moves) for moves in xy_s)
    predicted_movements = []

    for pos in range(max_length):
        positions = [xy_s[i][pos] for i in range(len(xy_s)) if pos < len(xy_s[i])]
        if len(positions) > 1:
            # positions = [(x1, y1), (x2, y2), ..., (x5, y5)]
            positions_array = np.array(positions)
            gmm = GaussianMixture(
                n_components=min(n_components, len(positions)), random_state=42
            )
            gmm.fit(positions_array)
            sampled_position, _ = gmm.sample(1)
            predicted_movements.append(sampled_position[0])
        elif len(positions) == 1:
            predicted_movements.append(positions[0])

    return np.array(predicted_movements)


def apply_movements_to_coordinates(df, query_uid, predicted_movements):
    last_coord = (
        df[(df["uid"] == query_uid) & (df["ts"] == END_TS)][["x", "y"]]
        .values[0]
        .astype(float)
    )
    results = []
    current_coord = last_coord.copy()
    for i, movement in enumerate(predicted_movements):
        current_coord += movement
        current_coord = np.clip(current_coord, 1, 200)
        ts = END_TS + 1 + i
        results.append(
            {
                "uid": query_uid,
                "ts": ts,
                "day": ts // 48 + 1,
                "time": ts % 48,
                "x": current_coord[0],
                "y": current_coord[1],
            }
        )
    return pd.DataFrame(results)


def calculate_user_similarity(train_uid, query_user):
    """Calculate similarity between query user and a single training user"""
    cur = []
    train_user = df[df["uid"] == train_uid]
    for day in range(1, 61):
        query_user_day = query_user[query_user["day"] == day].sort_values("ts")
        train_user_day = train_user[train_user["day"] == day].sort_values("ts")
        q_xy = get_xy_list_from_df_simple(query_user_day)
        t_xy = get_xy_list_from_df_simple(train_user_day)
        similarity = get_algo()((q_xy, t_xy))
        cur.append(similarity)

    # Keep only elements in 25-75 percentile and get mean
    cur = np.array(cur)
    q25, q75 = np.percentile(cur, [25, 75])
    filtered_similarities = cur[(cur >= q25) & (cur <= q75)]
    similarity = np.mean(filtered_similarities)

    return (similarity, train_uid)


def find_top_similar_users(query_uid, top_k=5):
    """Find top K most similar users using GEO-BLEU with multithreading"""
    query_user = df[df["uid"] == query_uid]
    train_users_max = 27000
    train_uids = list(range(1, train_users_max))

    similarities = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_uid = {
            executor.submit(calculate_user_similarity, train_uid, query_user): train_uid
            for train_uid in train_uids
        }

        for future in tqdm(
            as_completed(future_to_uid),
            total=len(train_uids),
            desc=f"Finding similar users for {query_uid}",
        ):
            train_uid = future_to_uid[future]
            try:
                similarity, uid = future.result()
                similarities.append((similarity, uid))
                print(f"Calculated: {similarity:.6f} for user {uid}")
            except Exception as exc:
                print(f"User {train_uid} generated an exception: {exc}")

    similarities.sort()
    return similarities[:top_k]


def visualize_user_paths(query_uid, similar_users, save_path=None):
    """
    Visualize the paths of the query user and its 5 most similar users

    Args:
        query_uid: The query user ID
        similar_users: List of (similarity_score, uid) tuples for similar users
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 10))

    # Define colors for different users
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

    # Plot query user path
    query_user_data = df[df["uid"] == query_uid]
    query_train_data = query_user_data[query_user_data["day"] <= settings.TRAIN_DAYS]
    query_xy = get_xy_list_from_df_simple(query_train_data.sort_values("ts"))

    if query_xy:
        x_coords, y_coords = zip(*query_xy)
        plt.plot(
            x_coords,
            y_coords,
            color=colors[0],
            linewidth=3,
            label=f"Query User {query_uid}",
            alpha=0.8,
        )
        plt.scatter(
            x_coords[0],
            y_coords[0],
            color=colors[0],
            s=100,
            marker="o",
            edgecolor="black",
            linewidth=2,
            label=f"Start {query_uid}",
        )
        plt.scatter(
            x_coords[-1],
            y_coords[-1],
            color=colors[0],
            s=100,
            marker="s",
            edgecolor="black",
            linewidth=2,
            label=f"End {query_uid}",
        )

    # Plot similar users' paths
    for i, (similarity, uid) in enumerate(similar_users[:5]):
        if i + 1 < len(colors):
            color = colors[i + 1]
            user_data = df[df["uid"] == uid]
            user_train_data = user_data[user_data["day"] <= settings.TRAIN_DAYS]
            user_xy = get_xy_list_from_df_simple(user_train_data.sort_values("ts"))

            if user_xy:
                x_coords, y_coords = zip(*user_xy)
                plt.plot(
                    x_coords,
                    y_coords,
                    color=color,
                    linewidth=2,
                    label=f"Similar User {uid} " f"(GEO-BLEU: {similarity:.4f})",
                    alpha=0.7,
                )
                plt.scatter(
                    x_coords[0],
                    y_coords[0],
                    color=color,
                    s=60,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                )
                plt.scatter(
                    x_coords[-1],
                    y_coords[-1],
                    color=color,
                    s=60,
                    marker="s",
                    edgecolor="black",
                    linewidth=1,
                )

    plt.xlim(settings.MIN_X, settings.MAX_X)
    plt.ylim(settings.MIN_Y, settings.MAX_Y)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title(f"Movement Paths: Query User {query_uid} and Top 5 Similar Users")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        save_file = f"user_{query_uid}_similarity_paths.png"
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_file}")

    plt.show()
    plt.close()


def main():
    # Load data
    global df
    print("Loading dataset...")
    df = pd.read_parquet("city_B_challengedata_converted.parquet")
    print(f"Loaded {len(df):,} rows")

    df["day"] = df["ts"] // 48 + 1
    df["time"] = df["ts"] % 48

    # Define users
    query_uids = list(range(27001, 27011))  # First 10 users
    all_predictions = []

    pred_df = df.copy()

    for query_uid in query_uids:
        print(f"\n=== Predicting for user {query_uid} ===")

        top_similar = find_top_similar_users(query_uid, top_k=5)

        print("Top 5 similar users:")
        for similarity, uid in top_similar:
            print(f"  User {uid}: GEO-BLEU = {similarity:.6f}")

        # Visualize the paths
        print("Creating visualization...")
        visualize_user_paths(query_uid, top_similar)

        print("Blending movements with GMM...")
        blended_movements = blend_extension_movements_with_gmm(
            [uid for _, uid in top_similar], n_components=2
        )
        print(f"Generated {len(blended_movements)} blended movements")

        # Convert to coordinates
        user_predictions = apply_movements_to_coordinates(
            df, query_uid, blended_movements
        )

        all_predictions.append(user_predictions)

    # Add all_predictions to pred_df
    pred_df = pd.concat([pred_df] + all_predictions, ignore_index=True)
    pred_df.to_parquet("predictions.parquet", index=False)

    all_predictions = pd.concat(all_predictions, ignore_index=True)
    all_predictions.to_csv("predictions.csv", index=False)


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
