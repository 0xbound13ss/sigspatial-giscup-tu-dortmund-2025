import pandas as pd
import time
from config import settings
import numpy as np
from utils import get_algo, get_xy_list_from_df


def truncate_by_ts(df):
    """
    Truncate DataFrame to the specified timestamp range.
    """
    start_ts = 0
    end_ts = settings.TRAIN_DAYS * settings.TIMESTAMPS_PER_DAY - 1  # should be 2879
    return df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)]


def get_global_mean_df(df):
    """
    Calculate global mean for x and y columns in the DataFrame.
    """
    # Calculate global mean
    mean_df = truncate_by_ts(df)
    mean_start_time = time.time()
    print("Calculating global mean for each column...")
    global_means = mean_df.mean(numeric_only=True)
    mean_time = time.time() - mean_start_time
    print(f"Global means calculated in {mean_time:.2f} seconds")
    print("Global means:")
    print(global_means)
    x_mean = global_means["x"]
    y_mean = global_means["y"]

    test_df = df[df["day"] > 60].copy()
    test_df["x"] = x_mean
    test_df["y"] = y_mean

    return test_df

    # Processed 1/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 2/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 3/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 4/3000 users... with GEO-BLEU score: 0.00000202
    # Processed 5/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 6/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 7/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 8/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 9/3000 users... with GEO-BLEU score: 0.00000004
    # Processed 10/3000 users... with GEO-BLEU score: 0.00000003
    # Processed 11/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 12/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 13/3000 users... with GEO-BLEU score: 0.00011658
    # Processed 14/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 15/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 16/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 17/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 18/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 19/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 20/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 21/3000 users... with GEO-BLEU score: 0.00000026


def get_global_mode_df(df):
    """
    Calculate global mode for x and y columns in the DataFrame.
    """
    # Calculate global mode
    mode_df = truncate_by_ts(df)
    mode_start_time = time.time()
    print("Calculating global mode for each column...")
    global_modes = mode_df.mode(numeric_only=True).iloc[0]
    mode_time = time.time() - mode_start_time
    print(f"Global modes calculated in {mode_time:.2f} seconds")
    print("Global modes:")
    print(global_modes)
    x_mode = global_modes["x"]
    y_mode = global_modes["y"]

    test_df = df[df["day"] > 60].copy()
    test_df["x"] = x_mode
    test_df["y"] = y_mode

    return test_df

    # Processed 1/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 2/3000 users... with GEO-BLEU score: 0.01086833
    # Processed 3/3000 users... with GEO-BLEU score: 0.00036305
    # Processed 4/3000 users... with GEO-BLEU score: 0.00001366
    # Processed 5/3000 users... with GEO-BLEU score: 0.07436818
    # Processed 6/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 7/3000 users... with GEO-BLEU score: 0.02557691
    # Processed 8/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 9/3000 users... with GEO-BLEU score: 0.02179367
    # Processed 10/3000 users... with GEO-BLEU score: 0.00000074
    # Processed 11/3000 users... with GEO-BLEU score: 0.00245091
    # Processed 12/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 13/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 14/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 15/3000 users... with GEO-BLEU score: 0.00023454


def get_per_user_mean_df(df):
    """
    Calculate per-user mean for x and y columns in the DataFrame.
    """
    # Calculate per-user mean
    user_means = df.groupby("uid")[["x", "y"]].mean().reset_index()
    user_means.rename(columns={"x": "mean_x", "y": "mean_y"}, inplace=True)

    test_df = df[df["day"] > 60].copy()
    test_df = test_df.merge(user_means, on="uid", how="left")
    test_df["x"] = test_df["mean_x"]
    test_df["y"] = test_df["mean_y"]
    return test_df

    # Processed 1/3000 users... with GEO-BLEU score: 0.00121135
    # Processed 2/3000 users... with GEO-BLEU score: 0.01171101
    # Processed 3/3000 users... with GEO-BLEU score: 0.00051079
    # Processed 4/3000 users... with GEO-BLEU score: 0.00000000
    # Processed 5/3000 users... with GEO-BLEU score: 0.00000149
    # Processed 6/3000 users... with GEO-BLEU score: 0.00294803
    # Processed 7/3000 users... with GEO-BLEU score: 0.00000086
    # Processed 8/3000 users... with GEO-BLEU score: 0.00554186
    # Processed 9/3000 users... with GEO-BLEU score: 0.04604767
    # Processed 10/3000 users... with GEO-BLEU score: 0.00000004
    # Processed 11/3000 users... with GEO-BLEU score: 0.00563905
    # Processed 12/3000 users... with GEO-BLEU score: 0.00010827
    # Processed 13/3000 users... with GEO-BLEU score: 0.00270977
    # Processed 14/3000 users... with GEO-BLEU score: 0.06045748
    # Processed 15/3000 users... with GEO-BLEU score: 0.00000017


def get_per_user_mode_df(df):
    """
    Calculate per-user mode for x and y columns in the DataFrame.
    """
    # Calculate per-user mode
    user_modes = (
        df.groupby("uid")[["x", "y"]].agg(lambda x: x.mode().iloc[0]).reset_index()
    )
    user_modes.rename(columns={"x": "mode_x", "y": "mode_y"}, inplace=True)

    test_df = df[df["day"] > 60].copy()
    test_df = test_df.merge(user_modes, on="uid", how="left")
    test_df["x"] = test_df["mode_x"]
    test_df["y"] = test_df["mode_y"]
    return test_df

    # Processed 1/3000 users... with GEO-BLEU score: 0.31735932
    # Processed 2/3000 users... with GEO-BLEU score: 0.31244202
    # Processed 3/3000 users... with GEO-BLEU score: 0.70328656
    # Processed 4/3000 users... with GEO-BLEU score: 0.44320573
    # Processed 5/3000 users... with GEO-BLEU score: 0.15570423
    # Processed 6/3000 users... with GEO-BLEU score: 0.40233142
    # Processed 7/3000 users... with GEO-BLEU score: 0.30131662
    # Processed 8/3000 users... with GEO-BLEU score: 0.00033613
    # Processed 9/3000 users... with GEO-BLEU score: 0.46256050
    # Processed 10/3000 users... with GEO-BLEU score: 0.22546528
    # Processed 11/3000 users... with GEO-BLEU score: 0.73642871
    # Processed 12/3000 users... with GEO-BLEU score: 0.51048673
    # Processed 13/3000 users... with GEO-BLEU score: 0.50608543
    # Processed 14/3000 users... with GEO-BLEU score: 0.44739682
    # Processed 15/3000 users... with GEO-BLEU score: 0.50742937
    # Processed 16/3000 users... with GEO-BLEU score: 0.46346104
    # Processed 17/3000 users... with GEO-BLEU score: 0.53503420
    # Processed 18/3000 users... with GEO-BLEU score: 0.16758883


def main():
    start_time = time.time()
    print("Loading the entire dataset...")
    df = pd.read_parquet("city_B_challengedata_converted.parquet")
    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    print(f"Loaded {len(df):,} rows")

    df = df.astype(
        {
            "uid": "uint16",  # uid <= 150000  # TODO: change to uint32 for dataset A
            "ts": "uint16",  # ts < 3600
            "x": "float32",  # float 1.0 <= x <= 200.0
            "y": "float32",  # float 1.0 <= y <= 200.0
        }
    )

    df["day"] = (
        df["ts"] // settings.TIMESTAMPS_PER_DAY + 1
    )  # Convert ts to day (1-indexed)
    df["time"] = df["ts"] % settings.TIMESTAMPS_PER_DAY  # Convert ts to time

    # Evaluate GEO-BLEU
    print("Calculating GEO-BLEU score...")
    # 1. Global mean
    # test_df = get_global_mean_df(df)
    # 2. Global mode
    # test_df = get_global_mode_df(df)
    # 3. Per-user mean
    # test_df = get_per_user_mean_df(df)
    # 4. Per-user mode
    test_df = get_per_user_mode_df(df)
    ref_df = df[df["day"] > 60]

    start_scores = time.time()
    print("Calculating GEO-BLEU scores for 3000 random users...")
    # 3000 random ints 1..27000
    scores = []
    for i, uid in enumerate(np.random.randint(1, 27001, 3000)):
        # Get user data for prediction and reference
        pred_user = get_xy_list_from_df(test_df, uid)
        ref_user = get_xy_list_from_df(ref_df, uid)
        print(f"len pred: {len(pred_user)}, len ref: {len(ref_user)}")

        # 1. Use optimized geobleu_by_day

        score = get_algo()(
            (pred_user, ref_user),
        )

        print(f"score: {score:.8f}")

        # 2. Use original calc_geobleu (commented out)
        # from geobleu_seq_eval import calc_geobleu
        # geobleu_score = calc_geobleu(
        #     sys_seq=pred_user[["day", "time", "x", "y"]].values.tolist(),
        #     ans_seq=ref_user[["day", "time", "x", "y"]].values.tolist(),
        #     processes=4,
        # )

        scores.append(score)
        print(f"Processed {i+1}/3000 users... with GEO-BLEU score: {score:.8f}")

    print(
        f"Calculated GEO-BLEU scores for {len(scores)} users in {time.time() - start_scores:.2f} seconds"
    )
    print(
        f"Min, max, avg GEO-BLEU scores: {min(scores):.8f}, {max(scores):.8f}, {np.mean(scores):.8f}"
    )

    # Processed 2993/3000 users... with GEO-BLEU score: 0.37151607
    # Processed 2994/3000 users... with GEO-BLEU score: 0.39986593
    # Processed 2995/3000 users... with GEO-BLEU score: 0.37245202
    # Processed 2996/3000 users... with GEO-BLEU score: 0.35179398
    # Processed 2997/3000 users... with GEO-BLEU score: 0.48795997
    # Processed 2998/3000 users... with GEO-BLEU score: 0.60129272
    # Processed 2999/3000 users... with GEO-BLEU score: 0.23229010
    # Processed 3000/3000 users... with GEO-BLEU score: 0.00000000
    # Calculated GEO-BLEU scores for 3000 users in 207.57 seconds
    # Min, max, avg GEO-BLEU scores: 0.00000000, 0.89646846, 0.37628995


if __name__ == "__main__":
    main()
