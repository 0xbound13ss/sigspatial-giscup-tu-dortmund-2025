import pandas as pd
import time


start_time = time.time()
print("Loading the entire dataset...")
df = pd.read_parquet("city_B_challengedata_converted.parquet")
load_time = time.time() - start_time
print(f"Data loaded in {load_time:.2f} seconds")
print(f"Loaded {len(df):,} rows")
