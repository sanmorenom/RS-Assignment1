import os
import copy
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import NCF


# ----------------------------
# Settings
# ----------------------------
SEED = 42
BATCH_SIZE = 256
EPOCHS = 20
PATIENCE = 3

#LEARNING_RATE = 0.001
#GMF_DIM = 64
#MLP_LAYERS = [64, 32, 16, 8]
#DROPOUT = 0.2

torch.manual_seed(SEED)


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
VAL_PATH = os.path.join(PROCESSED_DIR, "val.csv")
INFO_PATH = os.path.join(PROCESSED_DIR, "data_info.txt")
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_ncf_model.pt")
BEST_MODEL_INFO_PATH = os.path.join(CHECKPOINT_DIR, "best_model_info.txt")


# ----------------------------
# Dataset class
# ----------------------------
class InteractionDataset(Dataset):
    """
    Dataset for user-item-label interactions.
    """

    def __init__(self, dataframe):
        self.users = torch.tensor(dataframe["userId"].values, dtype=torch.long)
        self.items = torch.tensor(dataframe["movieId"].values, dtype=torch.long)
        self.labels = torch.tensor(dataframe["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]


# ----------------------------
# Helper function
# ----------------------------
def load_data_info(info_path):
    """
    Read num_users and num_items from data_info.txt
    """
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


def run_one_epoch(model, dataloader, criterion, optimizer, device, train=True):
    """
    Run one training or validation epoch.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    for users, items, labels in dataloader:
        users = users.to(device)
        items = items.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        preds = model(users, items)
        loss = criterion(preds, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * len(labels)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


# ----------------------------
# Main training
# ----------------------------
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    learning_rates = np.arange(1e-5,1e-3, 1.98e-4)
    gmf_dim_set = [16, 32, 64, 128]
    mlp_layer_sets = [[256, 128, 64, 32], [128, 64, 32, 16],[64, 32, 16, 8], [32, 16, 8, 4]]
    dropout_sets = np.arange(0.05,0.25, 0.5, dtype=float)
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    print("Train samples:", len(train_df))
    print("Validation samples:", len(val_df))

    # Load metadata
    num_users, num_items = load_data_info(INFO_PATH)
    print("Num users:", num_users)
    print("Num items:", num_items)

    # Create datasets
    train_dataset = InteractionDataset(train_df)
    val_dataset = InteractionDataset(val_df)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    best_val_losses = []

    for lr in learning_rates:
        for gmf_dim in gmf_dim_set:
            for mlp_layers in mlp_layer_sets:
                for dropout in dropout_sets:
                    
                    # Build model
                    model = NCF(
                        num_users=num_users,
                        num_items=num_items,
                        gmf_dim=gmf_dim,
                        mlp_layers=mlp_layers,
                        dropout=dropout
                    ).to(device)

                    # Loss and optimizer
                    criterion = nn.BCELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    best_epoch = 0
                    patience_counter = 0
                    best_val_loss = float("inf")

                    # Training loop
                    for epoch in range(1, EPOCHS + 1):
                        train_loss = run_one_epoch(
                            model=model,
                            dataloader=train_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device,
                            train=True
                        )

                        val_loss = run_one_epoch(
                            model=model,
                            dataloader=val_loader,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=device,
                            train=False
                        )
                        # Check improvement
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_epoch = epoch
                            patience_counter = 0
                        else:
                            patience_counter += 1

                        # Early stopping
                        if patience_counter >= PATIENCE:
                            break
                    
                    if len(best_val_losses)<5:
                        best_val_losses.append({"val_loss":val_loss,
                                                "epoch":epoch,
                                                "params":{"lr":lr,
                                                "gmf_dim":gmf_dim,
                                                "mlp_layers":mlp_layers,
                                                "dropout":dropout},
                                                "model":model.state_dict()})
                        best_val_losses = sorted(best_val_losses, key=lambda d: d['val_loss'])
                    elif best_val_losses[4]["val_loss"] > best_val_loss:
                        best_val_losses[4] = {"val_loss":val_loss,
                                                "epoch":epoch,
                                                "params":{"lr":lr,
                                                "gmf_dim":gmf_dim,
                                                "mlp_layers":mlp_layers,
                                                "dropout":dropout},
                                                "model":model.state_dict()}
                        best_val_losses = sorted(best_val_losses, key=lambda d: d['val_loss'])
        count = 1
        for x in best_val_losses:
            model_path = os.path.join(CHECKPOINT_DIR, f"best_ncf_model_{count}.pt")
            torch.save(x["model"].state_dict(), model_path)
            epoch = x["epoch"]
            val_loss = x["val_loss"]
            params = x["params"]
            with open(os.path.join(CHECKPOINT_DIR, f"best_model_info_{count}.txt"), "w", encoding="utf-8") as f:
                f.write(f"best_epoch={epoch}\n")
                f.write(f"best_val_loss={val_loss:.4f}\n")
                f.write(f"best_model_path={model_path}\n")
                f.writable(f"best_parameters={params}")
            count+=1
            print("Best epoch:", epoch)
            print("Best validation loss:", round(val_loss, 4))
            print("Best parameters:",params)
            print("Best model path:", model_path)
    # Save best model info
        

    print("Training finished.")
    


if __name__ == "__main__":
    main()