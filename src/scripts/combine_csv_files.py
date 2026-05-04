import pandas as pd

# Load both CSV files
df1 = pd.read_csv("data/processed/positive.csv")
df2 = pd.read_csv("data/processed/negative.csv")

# Concatenate them
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save to a new CSV
combined_df.to_csv("data/processed/combined.csv", index=False)