import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'BigramsModel',
    'OneLayerBigramModel',
]

class BigramsModel(nn.Module):
    def __init__(self, N: torch.Tensor, prior: int | float = 0):
        super().__init__()
        self.nalphabet = N.shape[0]
        self.N = N.clone().detach()
        N = N.to(dtype=torch.float32) + prior
        self.p = torch.nan_to_num((N / N.sum(1, keepdim=True)).log(), torch.nan, neginf=-1e6)

    def forward(self, x: torch.Tensor):
        return x.reshape(-1, self.nalphabet) @ self.p


class OneLayerBigramModel(nn.Module):
    def __init__(self, *, nalphabet: int, context_size: int):
        super().__init__()
        assert context_size == 1
        self.nalphabet = nalphabet
        self.w = nn.Parameter(torch.ones((nalphabet,nalphabet)) / nalphabet)
    
    def forward(self, x: torch.Tensor):
        return x.reshape(-1, self.nalphabet) @ self.w