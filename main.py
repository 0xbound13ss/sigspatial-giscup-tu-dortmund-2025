import pandas as pd
import glob
import os

filenames = [
    "city_A_challengedata.csv",
    "city_B_challengedata.csv",
    "city_C_challengedata.csv",
    "city_D_challengedata.csv",
]


def peek_csv_files():

    for file_path in filenames:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"\n{'='*80}\n{file_path} (Size: {file_size_mb:.2f} MB)\n{'='*80}")

        try:
            df = pd.read_csv(file_path, nrows=10)

            print(f"\nFirst 10 rows have {df.shape[1]} columns")

            print("\nColumns:")
            print(df.columns.tolist())

            print("\nFirst 5 rows:")
            print(df.head(5))

            with open(file_path, "r") as f:
                next(f)
                line_count = sum(1 for _ in f) + 1  # +1 for header
                print(f"\nTotal number of rows: approximately {line_count:,}")

        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")


if __name__ == "__main__":
    peek_csv_files()
