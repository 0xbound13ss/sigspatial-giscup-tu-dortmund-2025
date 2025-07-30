import pandas as pd
from config import settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import time
import os
from tqdm import tqdm

file_in = "city_A_challengedata.csv"  # Change this for different cities
file_out = "city_A_challengedata_converted.parquet"

sstart = time.time()

# Step 1: Convert CSV to temp parquet files (split into 10 files)
print("Step 1: Converting CSV to temp parquet files...")

chunk_size = 1_000_000
temp_files = []
chunk_num = 0
total_rows = 0

for chunk in pd.read_csv(file_in, dtype=int, chunksize=chunk_size):
    print(f"Processing chunk {chunk_num + 1}...")

    # Basic cleaning
    chunk["d"] = chunk["d"] - 1
    chunk = chunk[(chunk["x"] != 999) & (chunk["y"] != 999)]
    chunk = chunk.dropna(subset=["x", "y"])
    chunk["ts"] = chunk["d"] * 48 + chunk["t"]
    chunk = chunk.drop(["d", "t"], axis=1)

    chunk = chunk.astype(
        {
            "uid": "uint32",  # For city A
            "ts": "uint16",
            "x": "uint8",
            "y": "uint8",
        }
    )

    # Save to temp parquet file
    temp_file = f"temp_raw_{chunk_num:03d}.parquet"
    chunk.to_parquet(temp_file, index=False)
    temp_files.append(temp_file)
    total_rows += len(chunk)
    chunk_num += 1

    del chunk

print(f"Created {len(temp_files)} temp files with {total_rows:,} total rows")

# Step 2: Get all unique users from temp files
print("Step 2: Getting unique users...")
all_uids = set()
for temp_file in temp_files:
    chunk = pd.read_parquet(temp_file, columns=["uid"])
    all_uids.update(chunk["uid"].unique())
    del chunk

all_uids = sorted(list(all_uids))
print(f"Found {len(all_uids)} unique users")

# Step 3: Reorganize temp files by user ranges for efficient processing
print("Step 3: Reorganizing data by user ranges...")
start_time = time.time()

# Split users into groups and reorganize data accordingly
users_per_group = 5000  # Group 1000 users together
user_groups = []

for i in range(0, len(all_uids), users_per_group):
    user_group = all_uids[i : i + users_per_group]
    user_groups.append(user_group)

print(f"Split {len(all_uids)} users into {len(user_groups)} groups")

# For each user group, collect their data from all temp files
reorganized_files = []

for group_idx, user_group in enumerate(user_groups):
    print(
        f"Reorganizing group {group_idx + 1}/{len(user_groups)} (users {user_group[0]}-{user_group[-1]})"
    )

    user_set = set(user_group)
    group_data = []

    # Collect data for this user group from all temp files
    for temp_file in temp_files:
        chunk = pd.read_parquet(temp_file)
        group_chunk = chunk[chunk["uid"].isin(user_set)]
        if len(group_chunk) > 0:
            group_data.append(group_chunk)
        del chunk

    # Save reorganized data
    if group_data:
        group_df = pd.concat(group_data, ignore_index=True)
        reorganized_file = f"temp_group_{group_idx:03d}.parquet"
        group_df.to_parquet(reorganized_file, index=False)
        reorganized_files.append((reorganized_file, user_group))
        del group_df, group_data

print(f"Data reorganization completed in {time.time() - start_time:.2f} seconds")


# Step 4: Process users from reorganized files
def process_user_from_group(uid, group_data):
    """Process a single user from pre-loaded group data"""
    max_ts = settings.TIMESTAMPS_PER_DAY * settings.ALL_DAYS - 1

    # Get user data from group
    user_df = group_data[group_data["uid"] == uid].copy()
    if len(user_df) == 0:
        return None

    user_df = user_df.sort_values("ts")

    # Add missing start/end timestamps
    if user_df["ts"].min() != 0:
        first_record = user_df.iloc[0].copy()
        first_record["ts"] = 0
        user_df = pd.concat([pd.DataFrame([first_record]), user_df], ignore_index=True)

    if user_df["ts"].max() != max_ts:
        last_record = user_df.iloc[-1].copy()
        last_record["ts"] = max_ts
        user_df = pd.concat([user_df, pd.DataFrame([last_record])], ignore_index=True)

    # Forward fill missing timestamps
    complete_ts = pd.DataFrame({"uid": uid, "ts": range(0, max_ts + 1)})
    user_complete = complete_ts.merge(user_df, on=["uid", "ts"], how="left")
    user_complete["x"] = user_complete["x"].ffill()
    user_complete["y"] = user_complete["y"].ffill()

    return user_complete


print("Step 4: Processing users from reorganized files...")
start_time = time.time()

output_files = []
batch_num = 0

# Process each reorganized file
for reorganized_file, user_group in reorganized_files:
    print(f"Processing group file {batch_num + 1}/{len(reorganized_files)}")

    # Load the entire group data once
    group_data = pd.read_parquet(reorganized_file)

    # Process all users in this group
    batch_results = []
    for uid in tqdm(
        user_group, desc=f"Processing users in group {batch_num + 1}", leave=False
    ):
        result = process_user_from_group(uid, group_data)
        if result is not None:
            batch_results.append(result)

    # Save batch results
    if batch_results:
        batch_df = pd.concat(batch_results, ignore_index=True)
        output_file = f"temp_output_{batch_num:03d}.parquet"
        batch_df.to_parquet(output_file, index=False)
        output_files.append(output_file)
        del batch_df, batch_results

    del group_data
    batch_num += 1

process_time = time.time() - start_time
print(f"User processing completed in {process_time:.2f} seconds")

# Step 5: Combine output files into final parquet
print("Step 5: Combining output files into final parquet...")
save_start = time.time()

# Use pyarrow for efficient combination
import pyarrow.parquet as pq
import pyarrow as pa

# Read first output file to get schema
first_batch = pd.read_parquet(output_files[0])
schema = pa.Schema.from_pandas(first_batch)
writer = pq.ParquetWriter(file_out, schema, compression="snappy")

final_rows = 0

# Write each output file to final parquet
for i, output_file in enumerate(output_files):
    print(f"Writing output batch {i+1}/{len(output_files)}")

    batch = pd.read_parquet(output_file)
    table = pa.Table.from_pandas(batch, schema=schema)
    writer.write_table(table)
    final_rows += len(batch)
    del batch, table

writer.close()

save_time = time.time() - save_start
print(f"Final file saved in {save_time:.2f} seconds")

# Step 6: Cleanup temp files
print("Step 6: Cleaning up temp files...")
cleanup_files = temp_files + [f[0] for f in reorganized_files] + output_files
for temp_file in cleanup_files:
    try:
        os.remove(temp_file)
    except:
        pass

print(f"Cleanup completed")
print(f"Output file: {file_out}")
print(f"Total rows in final file: {final_rows:,}")
print(f"Total processing time: {time.time() - sstart:.2f} seconds")

# Show sample data
print("\nSample data for user 1:")
sample_df = pd.read_parquet(file_out, filters=[("uid", "=", 1)])
print(sample_df.head(10))
