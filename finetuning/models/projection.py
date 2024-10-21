import torch.nn as nn

class ProjectionBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)