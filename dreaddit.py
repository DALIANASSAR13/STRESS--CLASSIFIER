import pandas as pd

# ======================
# LOAD REDDIT DATASET
# ======================
reddit = pd.read_csv("dreaddit_labeled.csv")

# ======================
# CLEANING FUNCTION
# ======================
def clean_reddit(df):
    df = df.copy()

    # remove nulls
    df = df.dropna(subset=["text"])

    # remove very short text
    df = df[df["text"].str.split().str.len() >= 5]

    # remove duplicates
    df = df.drop_duplicates(subset=["text"])

    return df.reset_index(drop=True)

# ======================
# APPLY FILTERING
# ======================
reddit_filtered = clean_reddit(reddit)

print("After filtering:", len(reddit_filtered))

# ======================
# TARGET SIZE (from your design)
# ======================
reddit_target = 10780  # 30% of final dataset

# safety check (in case dataset is smaller)
reddit_target = min(reddit_target, len(reddit_filtered))

# ======================
# SAMPLING
# ======================
reddit_sample = reddit_filtered.sample(n=reddit_target, random_state=42)

print("Final Reddit size:", len(reddit_sample))

# ======================
# SAVE
# ======================
reddit_sample.to_csv("reddit_filtered.csv", index=False)