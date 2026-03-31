from model import NCF
import torch
import torch.nn as nn
import torch.optim as optim

model = NCF(
    num_users=6038,
    num_items=3533,
    gmf_dim=32,
    mlp_layers=[64, 32, 16, 8],
    dropout=0.2
)

users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
items = torch.tensor([10, 20, 30, 40], dtype=torch.long)
labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

pred_before = model(users, items)
loss_before = criterion(pred_before, labels)

optimizer.zero_grad()
loss_before.backward()
optimizer.step()

pred_after = model(users, items)
loss_after = criterion(pred_after, labels)

print("Loss before:", loss_before.item())
print("Loss after :", loss_after.item())