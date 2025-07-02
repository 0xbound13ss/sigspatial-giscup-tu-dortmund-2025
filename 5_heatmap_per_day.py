import pandas as pd
import time
from geobleu_seq_eval import calc_geobleu_humob25
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def optimize_day_data(df_filtered, target_day):
    """Pre-group day data by user for faster access."""
    grouped = {}
    for uid in range(1, 61):
        user_data = df_filtered[
            (df_filtered["uid"] == uid) & (df_filtered["day"] == target_day)
        ][["day", "time", "x", "y"]].values.tolist()
        grouped[uid] = user_data
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

    # Filter to first 60 users and first 11 days
    df_filtered = df[(df["uid"] <= 60) & (df["day"] <= 11)]

    DAYS = list(range(1, 12))

    for target_day in DAYS:
        start_time = time.time()
        print(f"Processing day {target_day}...")

        user_data = optimize_day_data(df_filtered, target_day)

        heatmap = np.zeros((60, 60))

        for i in range(60):
            heatmap[i, i] = 1.0

        tasks = []
        for user_y in range(1, 61):
            for user_z in range(user_y + 1, 61):
                tasks.append((user_y, user_z, user_data[user_y], user_data[user_z]))

        print(f"Day {target_day}: Calculating {len(tasks)} GEO-BLEU pairs...")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for user_y, user_z, data_y, data_z in tasks:
                print(f"data_y: {data_y}, data_z: {data_z}")
                future = executor.submit(calc_geobleu_humob25, (data_y, data_z))
                futures.append((future, user_y, user_z))

            completed = 0
            for future, user_y, user_z in futures:
                score = future.result()
                heatmap[user_y - 1, user_z - 1] = score
                heatmap[user_z - 1, user_y - 1] = score

                completed += 1
                completed_str = f"{completed}/{len(tasks)}"
                print(
                    f"Day {target_day}: Completed {completed_str} pairs: user {user_y} vs {user_z} - {score:.4f}"
                )

        elapsed = time.time() - start_time
        print(f"Day {target_day} heatmap completed in {elapsed:.2f} seconds")

        print("Heatmap:")
        print(heatmap)

        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.title(f"Day {target_day} - GEO-BLEU Heatmap")
        plt.xlabel("User Z")
        plt.ylabel("User Y")
        plt.colorbar(label="GEO-BLEU Score")
        plt.tight_layout()
        plt.savefig(
            f"day_{target_day}_geobleu_heatmap.png", dpi=150, bbox_inches="tight"
        )
        # plt.show()
