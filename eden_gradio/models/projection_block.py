import torch.nn as nn

class ProjectionBlock(nn.Module):
    def __init__(self, input_dim_CLIP, input_dim_phi2):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim_CLIP)
        self.proj = nn.Sequential(
            nn.Linear(input_dim_CLIP, input_dim_phi2),
            nn.GELU(),
            nn.Linear(input_dim_phi2, input_dim_phi2)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return self.proj(x)