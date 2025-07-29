import pandas as pd
import time
from utils import get_xy_list_from_df_simple, get_algo
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def optimize_user_data(user_df):
    """Pre-group user data by day for faster access."""
    grouped = {}
    for day in range(1, 61):
        day_data = get_xy_list_from_df_simple(user_df[user_df["day"] == day])
        grouped[day] = day_data
    return grouped


if __name__ == "__main__":
    start_time = time.time()
    print("Loading the entire dataset...")
    df = pd.read_parquet("city_B_challengedata_converted.parquet")
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Loaded {len(df):,} rows")

    df["day"] = df["ts"] // 48 + 1
    df["time"] = df["ts"] % 48

    filtered_df = df[(df["uid"] <= 60) & (df["day"] <= 60)]

    UIDS = list(range(30, 61))

    for uid in UIDS:
        user_df = filtered_df[filtered_df["uid"] == uid]

        start_time = time.time()
        print(f"Processing user {uid}...")

        day_data = optimize_user_data(user_df)

        heatmap = np.zeros((60, 60))

        for i in range(60):
            heatmap[i, i] = 1.0

        tasks = []
        for day_x in range(1, 61):
            for day_y in range(day_x + 1, 61):
                tasks.append((day_x, day_y, day_data[day_x], day_data[day_y]))

        print(f"User {uid}: Calculating {len(tasks)} GEO-BLEU pairs...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for day_x, day_y, data_x, data_y in tasks:
                future = executor.submit(
                    get_algo(),
                    (data_x, data_y),
                )
                futures.append((future, day_x, day_y))

            completed = 0
            for future, day_x, day_y in futures:
                score = future.result()
                heatmap[day_x - 1, day_y - 1] = score
                heatmap[day_y - 1, day_x - 1] = score

                completed += 1
                print(
                    f"User {uid}: Completed {completed}/{len(tasks)} pairs: day {day_x} vs {day_y} - {score:.4f}"
                )

        elapsed = time.time() - start_time
        print(f"User {uid} heatmap completed in {elapsed:.2f} seconds")

        print("Heatmap:")
        print(heatmap)

        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.title(f"User {uid} - GEO-BLEU Heatmap")
        plt.xlabel("Day Y")
        plt.ylabel("Day X")
        plt.colorbar(label="GEO-BLEU Score")
        plt.tight_layout()
        plt.savefig(f"user_{uid}_geobleu_heatmap.png", dpi=150, bbox_inches="tight")
        # plt.show()
