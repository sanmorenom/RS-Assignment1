import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('ratings.dat', delimiter='::', engine='python', header=None,
                 names=['userId', 'movieId', 'rating', 'timestamp'])

# Remove timestamp
df = df.iloc[:, 0:3]

# Change the ratings to binary
df["label"] = np.where(df["rating"] >= 4, 1, 0)


# Split data
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Negative sampling for train data
users = train['userId'].to_numpy()
items = train['movieId'].to_numpy()
labels = train['label'].to_numpy()  

all_items = train['movieId'].unique()

user_items = train.groupby('userId')['movieId'].apply(set).to_dict()
n_pos_per_user = train[train['label']==1].groupby('userId')['movieId'].count().to_dict()


neg_ratio = 3 #Change this ratio

neg_samples = []

for user, pos_count in n_pos_per_user.items():
    
    # number of negative sampling per user
    n_neg = pos_count * neg_ratio
    
    # Get movie list that has interaction with the user
    user_interacted = user_items[user] 
    
    # get candidate list with size two times of n_neg
    candidate_items = np.random.choice(all_items, size = n_neg*2, replace = True)
    
    # Filter candidate_items
    candidate_items = [item for item in candidate_items if item not in user_interacted]
    
    # Get candidate with n_neg size
    candidate_items = candidate_items[:n_neg]
    
    # Keep the final candidate
    for item in candidate_items:
        neg_samples.append((user, item, 0))
        

neg_df = pd.DataFrame(neg_samples, columns=['userId','movieId','label'])

# Embed to train data
train_final = pd.concat([train[['userId','movieId','label']], neg_df], ignore_index=True)

# Shuffle train_final
train_final = train_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("Number of label 1:", (train_final['label']==1).sum())
print("Number of label 0:", (train_final['label']==0).sum())

train_final.to_csv('train.csv', index = False)