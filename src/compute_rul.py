import pandas as pd
import os

# Create outputs folder
os.makedirs("outputs", exist_ok=True)

# Load cleaned train data
train_path = "outputs/train_FD001_clean.csv"  # from Day 1-2
train_df = pd.read_csv(train_path)

print("Train loaded:", train_df.shape)
print(train_df.head())

# Step 1: Find max cycle per engine
max_cycle_df = train_df.groupby("engine_id")["cycle"].max().reset_index()
max_cycle_df.columns = ["engine_id", "max_cycle"]

print("\nMax cycle table (first 5):")
print(max_cycle_df.head())

# Step 2: Merge max_cycle into train data
train_df = train_df.merge(max_cycle_df, on="engine_id", how="left")

# Step 3: Compute RUL
train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

# Drop max_cycle column (optional)
train_df.drop(columns=["max_cycle"], inplace=True)

print("\nAfter adding RUL:")
print(train_df.head())

# Step 4: Verify manually for 1 engine
engine_to_check = 1

engine_data = train_df[train_df["engine_id"] == engine_to_check].copy()

print(f"\n--- Verification for Engine {engine_to_check} ---")
print("First 5 rows:")
print(engine_data[["engine_id", "cycle", "RUL"]].head())

print("\nLast 5 rows:")
print(engine_data[["engine_id", "cycle", "RUL"]].tail())

print("\nCheck:")
print("RUL at last cycle should be 0")
print("Last cycle RUL:", engine_data["RUL"].iloc[-1])

# Step 5: Save final train with RUL
train_df.to_csv("outputs/train_FD001_with_RUL.csv", index=False)
print("\nSaved: outputs/train_FD001_with_RUL.csv")