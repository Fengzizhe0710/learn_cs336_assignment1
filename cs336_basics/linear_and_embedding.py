import torch
from torch import nn

class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, weights: torch.Tensor, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        
        if weights is not None:
            self.W = weights
        else:
            self.W = nn.Parameter(torch.empty(self.out_features, self.in_features, device=self.device, dtype=self.dtype))

            std = 2 / (self.in_features + self.out_features) ** 0.5
            torch.nn.init.trunc_normal_(self.W, std=std, a = -3 * std, b = 3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T
