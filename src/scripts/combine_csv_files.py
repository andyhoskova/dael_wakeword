import pandas as pd

# Load CSV files
df1 = pd.read_csv("src/data/processed/negative.csv")
df2 = pd.read_csv("src/data/processed/negative_combined.csv")
#df3 = pd.read_csv("src/data/post_augmentation/negative_similar_words_raw.csv")

# Concatenate them
combined_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicate filenames, keeping the first occurrence
combined_df = combined_df.drop_duplicates(subset=["filename"], keep="first")

# Save to a new CSV
combined_df.to_csv("src/data/processed/negative_new.csv", index=False)