import pandas as pd
import numpy as np

# 1) Load dataset
train_path = "data/train_FD001.txt"
test_path = "data/test_FD001.txt"
rul_path = "data/RUL_FD001.txt"

# Read train
train_df = pd.read_csv(train_path, sep=" ", header=None)
train_df = train_df.dropna(axis=1, how="all")  # remove empty columns

# Create column names
columns = ["engine_id", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
train_df.columns = columns

print("Train data loaded successfully!")
print("Train shape:", train_df.shape)
print(train_df.head())

# 2) Check missing values
print("\nTotal missing values:", train_df.isnull().sum().sum())

# 3) Print required info
num_engines = train_df["engine_id"].nunique()
print("\nNumber of engines:", num_engines)

max_cycle_each_engine = train_df.groupby("engine_id")["cycle"].max()
print("\nMax cycle for each engine (first 10):")
print(max_cycle_each_engine.head(10))

print("\nOverall maximum cycle:", max_cycle_each_engine.max())

sensor_cols = [c for c in train_df.columns if c.startswith("s")]
print("\nNumber of sensors:", len(sensor_cols))

# Load test and RUL also
test_df = pd.read_csv(test_path, sep=" ", header=None)
test_df = test_df.dropna(axis=1, how="all")
test_df.columns = columns

rul_df = pd.read_csv(rul_path, header=None)
rul_df.columns = ["RUL"]

print("\nTest shape:", test_df.shape)
print("RUL shape:", rul_df.shape)

# Save cleaned csv
train_df.to_csv("outputs/train_FD001_clean.csv", index=False)
test_df.to_csv("outputs/test_FD001_clean.csv", index=False)

print("\nSaved cleaned CSV files inside outputs/ folder.")
