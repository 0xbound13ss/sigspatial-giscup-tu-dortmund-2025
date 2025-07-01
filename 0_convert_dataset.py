import pandas as pd
from config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time

file_in = "city_B_challengedata.csv"
file_out = "city_B_challengedata_converted.csv"

sstart = time.time()

# columns: uid,d,t,x,y
# 1. Load original .csv
# Filter out 999 x,y values
# Convert to ts and drop d,t columns
df = pd.read_csv(file_in, dtype=int)
df["d"] = df["d"] - 1
df = df[(df["x"] != 999) & (df["y"] != 999)]
df["ts"] = df["d"] * 48 + df["t"]
df = df.drop(["d", "t"], axis=1)

df = df.astype(
    {
        "uid": "uint16",  # uid <= 150000  # TODO: change to uint132 for dataset A
        "ts": "uint16",  # ts < 3600
        "x": "uint8",  # 1 <= x <= 200
        "y": "uint8",  # 1 <= y <= 200
    }
)

print(f"Adding missing timestamps for users in {file_in}...")
start_time = time.time()


# 2. Add ts=0 and ts=3599 points for train users:
max_ts = settings.TIMESTAMPS_PER_DAY * settings.ALL_DAYS - 1  # should be 3599
train_users = range(1, df["uid"].max() - settings.TEST_USERS + 1)
train_df = df[df["uid"].isin(train_users)]

# - Get the last record for each user
# - Filter users whose last ts is not max_ts
last_records = train_df.loc[train_df.groupby("uid")["ts"].idxmax()]
users_need_max_ts = last_records[last_records["ts"] != max_ts]
if not users_need_max_ts.empty:
    # Copy last ts and replace with max_ts, insert back into df
    new_records = users_need_max_ts.copy()
    new_records["ts"] = max_ts
    df = pd.concat([df, new_records], ignore_index=True)

# Same for ts=0
first_records = train_df.loc[train_df.groupby("uid")["ts"].idxmin()]
users_need_min_ts = first_records[first_records["ts"] != 0]
if not users_need_min_ts.empty:
    # Copy first ts and replace with ts=0, insert back into df
    new_records = users_need_min_ts.copy()
    new_records["ts"] = 0
    df = pd.concat([df, new_records], ignore_index=True)

print(f"Missing timestamps added - Elapsed: {time.time() - start_time:.2f}s")


# 3. Propagate missing ts for 0..max_ts for each user (multi-threaded)
def process_user(uid, user_df, max_ts):
    """Process a single user's data to fill missing timestamps."""
    complete_ts = pd.DataFrame({"uid": uid, "ts": range(0, max_ts + 1)})
    # Merge with existing data and forward fill
    user_complete = complete_ts.merge(user_df, on=["uid", "ts"], how="left")
    user_complete["x"] = user_complete["x"].ffill()
    user_complete["y"] = user_complete["y"].ffill()
    return user_complete


df = df.sort_values(by=["uid", "ts"])

# Group data by user
user_groups = df.groupby("uid")
unique_uids = df["uid"].unique()

print(f"Starting ffill for {len(unique_uids)} users with multi-threading...")
start_time = time.time()

user_dfs = []
with ThreadPoolExecutor(max_workers=8) as executor:
    future_to_uid = {
        executor.submit(process_user, uid, group.copy(), max_ts): uid
        for uid, group in user_groups
    }
    for i, future in enumerate(as_completed(future_to_uid)):
        uid = future_to_uid[future]
        try:
            result = future.result()
            user_dfs.append(result)
            if i % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Processed {i+1}/{len(unique_uids)} users - Elapsed: {elapsed:.2f}s"
                )
        except Exception as exc:
            print(f"User {uid} generated an exception: {exc}")

df = pd.concat(user_dfs, ignore_index=True)
total_time = time.time() - start_time
# For dataset B: Multi-threading completed in 28.92 seconds
print(f"Multi-threading completed in {total_time:.2f} seconds")
print(f"Average time per user: {total_time/len(unique_uids):.4f}s")

# Show df for user 1
print(df[df["uid"] == 1])

print("Sorting data by uid, ts...")
sort_start = time.time()
df = df.sort_values(by=["uid", "ts"])
print(f"Sorting completed in {time.time() - sort_start:.2f} seconds")


# 4. Save the processed dataframe to Parquet format
print(f"Saving processed data to {file_out}...")
save_start = time.time()

parquet_file = file_out.replace(".csv", ".parquet")
df.to_parquet(parquet_file, index=False, compression="snappy")

save_time = time.time() - save_start
print(f"Data saved to Parquet in {save_time:.2f} seconds")
print(f"Parquet file: {parquet_file}")

# Or to save as .csv
# df.to_csv(
#     file_out,
#     index=False,
#     columns=["uid", "ts", "x", "y"],
#     float_format="%.0f",  # No decimal places for integers
#     lineterminator="\n",  # Unix line endings
# )
# save_time = time.time() - save_start
# print(f"Data saved in {save_time:.2f} seconds")
# print(f"Output file: {file_out}")
# print(f"Total rows: {len(df):,}")
# print(f"File size: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB in memory")

print(f"Total processing time: {time.time() - sstart:.2f} seconds")
