import torch
import torch.nn as nn

class EntropyEmbeddingProjection(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.entropy_projection = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.SiLU(),
            # nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, hidden_size),
            # nn.LayerNorm(hidden_size)
        )
        
        # Initialize with Xavier initialization scaled down for stable start
        for module in self.entropy_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)  # Very small scale
                nn.init.normal_(module.bias, mean=0.0, std=0.1)

    def forward(self, entropy: torch.Tensor) -> torch.Tensor:
        # entropy: (batch_size, seq_len)
        # output: (batch_size, seq_len, hidden_size)
        entropy_expanded = entropy.unsqueeze(-1)  # (batch_size, seq_len, 1)
        return self.entropy_projection(entropy_expanded)