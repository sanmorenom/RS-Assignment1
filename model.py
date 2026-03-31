import torch
import torch.nn as nn


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model.
    It combines:
    1. GMF branch
    2. MLP branch
    3. Final fusion layer
    """

    def __init__(self, num_users, num_items, gmf_dim=32, mlp_layers=None, dropout=0.0):
        super().__init__()

        if mlp_layers is None:
            mlp_layers = [64, 32, 16, 8]

        if mlp_layers[0] % 2 != 0:
            raise ValueError("The first MLP layer size must be even.")

        # Save settings
        self.num_users = num_users
        self.num_items = num_items
        self.gmf_dim = gmf_dim
        self.mlp_layers = mlp_layers

        # GMF embeddings
        self.user_embedding_gmf = nn.Embedding(num_users, gmf_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, gmf_dim)

        # MLP embeddings
        # Example: if mlp_layers[0] = 64, then user emb = 32 and item emb = 32
        mlp_embedding_dim = mlp_layers[0] // 2
        self.user_embedding_mlp = nn.Embedding(num_users, mlp_embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, mlp_embedding_dim)

        # Build MLP layers
        mlp_modules = []
        for in_dim, out_dim in zip(mlp_layers[:-1], mlp_layers[1:]):
            mlp_modules.append(nn.Linear(in_dim, out_dim))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*mlp_modules)

        # Final layer after combining GMF and MLP
        fusion_dim = gmf_dim + mlp_layers[-1]
        self.output_layer = nn.Linear(fusion_dim, 1)

        # Sigmoid for final probability
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        nn.init.normal_(self.user_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, std=0.01)
        nn.init.normal_(self.user_embedding_mlp.weight, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, std=0.01)

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, user_ids, item_ids):
        """
        user_ids: tensor of shape [batch_size]
        item_ids: tensor of shape [batch_size]

        Returns:
            probabilities of shape [batch_size]
        """

        # ----- GMF branch -----
        user_gmf = self.user_embedding_gmf(user_ids)
        item_gmf = self.item_embedding_gmf(item_ids)

        # Element-wise product
        gmf_output = user_gmf * item_gmf

        # ----- MLP branch -----
        user_mlp = self.user_embedding_mlp(user_ids)
        item_mlp = self.item_embedding_mlp(item_ids)

        # Concatenate user and item embeddings
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)

        # Pass through MLP
        mlp_output = self.mlp(mlp_input)

        # ----- Fusion -----
        combined = torch.cat([gmf_output, mlp_output], dim=1)

        # Final prediction
        logits = self.output_layer(combined).squeeze(-1)
        probs = self.sigmoid(logits)

        return probs