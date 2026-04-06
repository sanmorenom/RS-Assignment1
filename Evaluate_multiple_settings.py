import os
import math
import numpy as np
import pandas as pd
import torch
import ast
from model import NCF


# ----------------------------
# Settings
# ----------------------------
SEED = 42
TOP_K = 10
PRED_BATCH_SIZE = 2048

torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

INFO_PATH = os.path.join(PROCESSED_DIR, "data_info.txt")
TRAIN_POS_PATH = os.path.join(PROCESSED_DIR, "train_positive.csv")
VAL_POS_PATH = os.path.join(PROCESSED_DIR, "val_positive.csv")
TEST_POS_PATH = os.path.join(PROCESSED_DIR, "test_positive.csv")
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_ncf_model.pt")
BEST_MODEL_INFO_PATH = os.path.join(CHECKPOINT_DIR, "best_model_info.txt")


# ----------------------------
# Helper functions
# ----------------------------
def load_data_info(info_path):
    """Read num_users and num_items from data_info.txt"""
    info = {}

    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            info[key] = value

    num_users = int(info["num_users"])
    num_items = int(info["num_items"])

    return num_users, num_items


def load_best_model_info(info_path):
    """Read best epoch and best validation loss from best_model_info.txt"""
    info = {}

    if not os.path.exists(info_path):
        return None

    with open(info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            info[key] = value

    return info


def build_user_item_dict(df):
    """Convert a dataframe into: userId -> set(movieId)"""
    return df.groupby("userId")["movieId"].apply(set).to_dict()


def recall_at_k(recommended_items, ground_truth_items, k=10):
    """Compute Recall@K for one user"""
    recommended_items = recommended_items[:k]
    hits = sum(1 for item in recommended_items if item in ground_truth_items)
    return hits / len(ground_truth_items)


def ndcg_at_k(recommended_items, ground_truth_items, k=10):
    """Compute NDCG@K for one user"""
    recommended_items = recommended_items[:k]

    dcg = 0.0
    for rank, item in enumerate(recommended_items):
        if item in ground_truth_items:
            dcg += 1.0 / math.log2(rank + 2)

    ideal_hits = min(len(ground_truth_items), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def predict_scores_for_user(model, user_id, candidate_items, device, batch_size=2048):
    """Predict scores for one user over many candidate items"""
    scores = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(candidate_items), batch_size):
            batch_items = candidate_items[start:start + batch_size]

            user_tensor = torch.full(
                (len(batch_items),),
                user_id,
                dtype=torch.long,
                device=device
            )
            item_tensor = torch.tensor(batch_items, dtype=torch.long, device=device)

            batch_scores = model(user_tensor, item_tensor)
            scores.extend(batch_scores.cpu().numpy())

    return np.array(scores)


# ----------------------------
# Main evaluation
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load metadata
    num_users, num_items = load_data_info(INFO_PATH)
    print("Num users:", num_users)
    print("Num items:", num_items)

    for i in range(1,6):
    # Show which best model is being evaluated
        best_model_info_path = os.path.join(CHECKPOINT_DIR, f"best_model_info_{i}.txt")
        best_model_path = os.path.join(CHECKPOINT_DIR, f"best_ncf_model_{i}.pt")
        best_model_info = load_best_model_info(best_model_info_path)
        best_model_gmf_dmi = ast.literal_eval(best_model_info.get("best_gmf_dim", "unknown"))
        best_model_mlp_layers = ast.literal_eval(best_model_info.get("best_mlp_layers", "unknown"))
        best_model_dropout = float(best_model_info.get("best_dropout", "unknown"))
        if best_model_info is not None:
            print("Best model info found:")
            print("Best epoch:", best_model_info.get("best_epoch", "unknown"))
            print("Best validation loss:", best_model_info.get("best_val_loss", "unknown"))
        else:
            print("best_model_info.txt not found.")

        # Load positive interactions
        train_pos = pd.read_csv(TRAIN_POS_PATH)
        val_pos = pd.read_csv(VAL_POS_PATH)
        test_pos = pd.read_csv(TEST_POS_PATH)

        # Build dictionaries
        train_user_items = build_user_item_dict(train_pos)
        val_user_items = build_user_item_dict(val_pos)
        test_user_items = build_user_item_dict(test_pos)

        # Build seen items = train + val
        seen_user_items = {}
        for user in range(num_users):
            train_items = train_user_items.get(user, set())
            val_items = val_user_items.get(user, set())
            seen_user_items[user] = train_items.union(val_items)

        all_items = set(range(num_items))

        # Build model
        model = NCF(
            num_users=num_users,
            num_items=num_items,
            gmf_dim=best_model_gmf_dmi,
            mlp_layers=best_model_mlp_layers,
            dropout=best_model_dropout
        ).to(device)

        # Load best trained weights
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        print("Loaded model from:", best_model_path)
        print("Start full-ranking evaluation...")

        recalls = []
        ndcgs = []
        evaluated_users = 0

        users_to_evaluate = sorted(test_user_items.keys())

        for idx, user in enumerate(users_to_evaluate, start=1):
            ground_truth = set(test_user_items[user])

            seen_items = seen_user_items.get(user, set())
            ground_truth = ground_truth - seen_items

            if len(ground_truth) == 0:
                continue

            candidate_items = list(all_items - seen_items)

            if len(candidate_items) == 0:
                continue

            scores = predict_scores_for_user(
                model=model,
                user_id=user,
                candidate_items=candidate_items,
                device=device,
                batch_size=PRED_BATCH_SIZE
            )

            if len(candidate_items) <= TOP_K:
                top_k_indices = np.argsort(scores)[::-1]
            else:
                top_k_indices = np.argpartition(scores, -TOP_K)[-TOP_K:]
                top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

            recommended_items = [candidate_items[i] for i in top_k_indices]

            user_recall = recall_at_k(recommended_items, ground_truth, k=TOP_K)
            user_ndcg = ndcg_at_k(recommended_items, ground_truth, k=TOP_K)

            recalls.append(user_recall)
            ndcgs.append(user_ndcg)
            evaluated_users += 1

            
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0

        print(f"\nEvaluation finished for model {i}.")
        print("Evaluated users:", evaluated_users)
        print(f"Recall@{TOP_K}: {mean_recall:.4f}")
        print(f"NDCG@{TOP_K}: {mean_ndcg:.4f}")


if __name__ == "__main__":
    main()