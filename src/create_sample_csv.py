import pandas as pd

# Load a few rows from your existing dataset
df_sample = pd.read_csv("data/KDDTrain+.txt", header=None).iloc[:100]  # first 100 rows
df_sample.to_csv("data/new_traffic.csv", index=False, header=False)

print("Sample CSV saved as new_traffic.csv")
