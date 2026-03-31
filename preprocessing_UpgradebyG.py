import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

SEED = 42
np.random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Load MovieLens 1M
# ----------------------------
ratings_path = os.path.join(BASE_DIR, "ml-1m", "ratings.dat")

df = pd.read_csv(
    ratings_path,
    delimiter="::",
    engine="python",
    header=None,
    names=["userId", "movieId", "rating", "timestamp"]
)

# ----------------------------
# Keep only positive interactions
# rating >= 4 => positive implicit feedback
# ----------------------------
df = df[["userId", "movieId", "rating"]].copy()
df = df[df["rating"] >= 4].copy()
df["label"] = 1
df = df[["userId", "movieId", "label"]]

# ----------------------------
# Remap ids to consecutive indices
# ----------------------------
user2idx = {u: i for i, u in enumerate(df["userId"].unique())}
item2idx = {m: i for i, m in enumerate(df["movieId"].unique())}

df["userId"] = df["userId"].map(user2idx)
df["movieId"] = df["movieId"].map(item2idx)

num_users = df["userId"].nunique()
num_items = df["movieId"].nunique()

# ----------------------------
# Split 70 / 15 / 15
# ----------------------------
train, temp = train_test_split(df, test_size=0.30, random_state=SEED)
val, test = train_test_split(temp, test_size=0.50, random_state=SEED)

# Reset indices for cleaner saved files
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)

# ----------------------------
# All items in dataset
# ----------------------------
all_items = set(df["movieId"].unique())

# Use all positive interactions of each user
user_items_all = df.groupby("userId")["movieId"].apply(set).to_dict()

def negative_sampling(pos_df, user_items_all, all_items, neg_ratio=3):
    """
    Generate negative samples for a dataframe of positive interactions.
    For each positive interaction, sample 'neg_ratio' items
    not interacted with by the same user.
    """
    neg_samples = []

    for user, group in pos_df.groupby("userId"):
        n_pos = len(group)
        n_neg = n_pos * neg_ratio

        interacted = user_items_all[user]
        candidates = list(all_items - interacted)

        if len(candidates) == 0:
            continue

        replace_flag = len(candidates) < n_neg
        sampled_neg = np.random.choice(candidates, size=n_neg, replace=replace_flag)

        for item in sampled_neg:
            neg_samples.append((user, item, 0))

    neg_df = pd.DataFrame(neg_samples, columns=["userId", "movieId", "label"])
    return neg_df

# ----------------------------
# Negative samples for train and validation
# ----------------------------
train_neg = negative_sampling(train, user_items_all, all_items, neg_ratio=3)
val_neg = negative_sampling(val, user_items_all, all_items, neg_ratio=3)

# ----------------------------
# Final datasets
# ----------------------------
train_final = pd.concat([train[["userId", "movieId", "label"]], train_neg], ignore_index=True)
val_final = pd.concat([val[["userId", "movieId", "label"]], val_neg], ignore_index=True)

train_final = train_final.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_final = val_final.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ----------------------------
# Save files inside RS-Assignment1/processed
# ----------------------------
processed_dir = os.path.join(BASE_DIR, "processed")
os.makedirs(processed_dir, exist_ok=True)

# Main splits
train_final.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
val_final.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
test.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

# Also save pure positive splits (useful for debugging/evaluation)
train.to_csv(os.path.join(processed_dir, "train_positive.csv"), index=False)
val.to_csv(os.path.join(processed_dir, "val_positive.csv"), index=False)
test.to_csv(os.path.join(processed_dir, "test_positive.csv"), index=False)

# ----------------------------
# Save mappings
# ----------------------------
user_mapping_df = pd.DataFrame(
    list(user2idx.items()),
    columns=["original_userId", "mapped_userId"]
).sort_values("mapped_userId")

item_mapping_df = pd.DataFrame(
    list(item2idx.items()),
    columns=["original_movieId", "mapped_movieId"]
).sort_values("mapped_movieId")

user_mapping_df.to_csv(os.path.join(processed_dir, "user_mapping.csv"), index=False)
item_mapping_df.to_csv(os.path.join(processed_dir, "item_mapping.csv"), index=False)

# ----------------------------
# Save user histories for later evaluation
# Long format: one row per (userId, movieId)
# ----------------------------
train_user_history = train[["userId", "movieId"]].copy().sort_values(["userId", "movieId"])
all_user_history = df[["userId", "movieId"]].copy().sort_values(["userId", "movieId"])

train_user_history.to_csv(os.path.join(processed_dir, "train_user_history.csv"), index=False)
all_user_history.to_csv(os.path.join(processed_dir, "all_user_history.csv"), index=False)

# Optional grouped histories (easy to inspect manually)
train_user_history_grouped = (
    train.groupby("userId")["movieId"]
    .apply(list)
    .reset_index()
    .rename(columns={"movieId": "train_items"})
)

all_user_history_grouped = (
    df.groupby("userId")["movieId"]
    .apply(list)
    .reset_index()
    .rename(columns={"movieId": "all_items"})
)

train_user_history_grouped.to_csv(
    os.path.join(processed_dir, "train_user_history_grouped.csv"),
    index=False
)

all_user_history_grouped.to_csv(
    os.path.join(processed_dir, "all_user_history_grouped.csv"),
    index=False
)

# ----------------------------
# Save metadata
# ----------------------------
with open(os.path.join(processed_dir, "data_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"num_users={num_users}\n")
    f.write(f"num_items={num_items}\n")
    f.write(f"train_positives={len(train)}\n")
    f.write(f"train_negatives={len(train_neg)}\n")
    f.write(f"val_positives={len(val)}\n")
    f.write(f"val_negatives={len(val_neg)}\n")
    f.write(f"test_positives={len(test)}\n")
    f.write("negative_sampling_ratio=3\n")
    f.write("implicit_feedback_threshold=rating>=4\n")

# ----------------------------
# Print summary
# ----------------------------
print("Train positives:", (train_final["label"] == 1).sum())
print("Train negatives:", (train_final["label"] == 0).sum())
print("Val positives:", (val_final["label"] == 1).sum())
print("Val negatives:", (val_final["label"] == 0).sum())
print("Test positives:", len(test))
print("Num users:", num_users)
print("Num items:", num_items)
print("Saved in:", processed_dir)