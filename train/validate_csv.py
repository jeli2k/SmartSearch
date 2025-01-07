import pandas as pd

# Path to your train_data.csv
file_path = 'train/train_data.csv'

# Load the CSV without skipping bad lines
try:
    df = pd.read_csv(file_path, on_bad_lines='skip', dtype=str)
except Exception as e:
    print(f"Error reading CSV: {e}")
else:
    print(f"Total rows: {len(df)}")
    print("Checking for NaN values...")
    print(df.isna().sum())  # Count NaN values per column
