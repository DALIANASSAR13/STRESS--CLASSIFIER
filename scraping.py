import pandas as pd

# ======================
# Load original scraped dataset
# ======================
df = pd.read_csv("final_merged_stress_Scraping.csv")  
# (غيري الاسم حسب ملفك)

# ======================
# Basic filtering function
# ======================
def clean_dataset(df):
    df = df.copy()

    # remove missing values
    df = df.dropna(subset=["text"])

    # remove very short comments
    df = df[df["text"].str.split().str.len() >= 5]

    # remove duplicates
    df = df.drop_duplicates(subset=["text"])

    # remove empty strings
    df = df[df["text"].str.strip() != ""]

    return df.reset_index(drop=True)

# ======================
# Apply filtering
# ======================
filtered_df = clean_dataset(df)


filtered_df.to_csv("youtube_scraped_filtered.csv", index=False)

print("Done ✔ Filtered dataset saved successfully")
print("Original size:", len(df))
print("Filtered size:", len(filtered_df))
print("Original:")
print(df["label"].value_counts())

print("\nFiltered:")
print(filtered_df["label"].value_counts())