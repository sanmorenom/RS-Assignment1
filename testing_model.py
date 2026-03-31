from model import NCF
import torch

model = NCF(
    num_users=6038,
    num_items=3533,
    gmf_dim=32,
    mlp_layers=[64, 32, 16, 8],
    dropout=0.2
)

print(model)

users = torch.tensor([0, 1, 2, 3], dtype=torch.long)
items = torch.tensor([10, 20, 30, 40], dtype=torch.long)

outputs = model(users, items)

print("Output shape:", outputs.shape)
print("Outputs:", outputs)



