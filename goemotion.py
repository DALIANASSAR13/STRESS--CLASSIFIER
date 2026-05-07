import pandas as pd

# ======================
# LOAD DATASET
# ======================
goemo = pd.read_csv("goemotions_labeled_dataset.csv")

# ======================
# BASIC FILTERING
# ======================
def clean_goemo(df):
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
goemo_filtered = clean_goemo(goemo)

print("After filtering:", len(goemo_filtered))

# ======================
# TAKE 30%
# ======================
goemo_30 = goemo_filtered.sample(n= 3531, random_state=42)

print("30% size:", len(goemo_30))

# ======================
# SAVE (optional)
# ======================
goemo_30.to_csv("goemotions_filtered.csv", index=False)