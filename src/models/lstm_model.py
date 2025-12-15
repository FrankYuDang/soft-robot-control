import torch
import torch.nn as nn
from typing import Dict, Any

class AttentionLayer(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Query: Last time step [batch, 1, hidden]
        query = x[:, -1:, :]
        # Key & Value: Full sequence
        key = x
        value = x
        
        attn_output, _ = self.mha(query, key, value)
        output = self.layer_norm(query + attn_output)
        return output.squeeze(1)

class AttnLSTM(nn.Module):
    """
    The main architecture combining LSTM with Attention.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 256, 
        num_layers: int = 2,
        output_dim: int = 3,
        num_heads: int = 4
    ):
        super(AttnLSTM, self).__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "num_heads": num_heads
        }

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attention = AttentionLayer(hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        predictions = self.fc(attn_out)
        return predictions