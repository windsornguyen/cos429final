import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = x + self.attn(x, x, x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        x = x + residual
        return x


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        residual = x
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = x + residual
        return x


class ASLInterpreter(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_classes,
        dropout=0.25,
        max_length=500,
    ):
        super().__init__()
        self.max_length = max_length
        self.d_model = embed_dim

        # Graph-based positional encoding layers
        self.graph_conv = nn.ModuleList(
            [
                GraphConvolution(4, embed_dim),
                GraphConvolution(embed_dim, embed_dim),
            ]
        )

        # Learned positional embedding layer
        self.pos_embedding = nn.Embedding(max_length, embed_dim)

        self.transformer = nn.Sequential(
            *(
                AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout)
                for _ in range(num_layers)
            )
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrix):
        # Graph-based positional encoding
        for graph_layer in self.graph_conv:
            x = graph_layer(x, adj_matrix)
            x = F.relu(x)

        # Add learned positional embeddings
        position_ids = torch.arange(self.max_length, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.pos_embedding(position_ids)
        x = x + position_embeddings

        # Apply transformer layers
        x = self.dropout(x)
        x = self.transformer(x)

        # Mean pooling and MLP head
        x = x.mean(dim=1)
        x = self.mlp_head(x)

        return x
