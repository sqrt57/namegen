import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'BigramsModel',
    'OneLayerBigramModel',
    'ProbabilisticEmbeddingModel',
]

class BigramsModel(nn.Module):
    def __init__(self, N: torch.Tensor, prior: int | float = 0, *, nalphabet=None, context_size=1):
        super().__init__()
        if nalphabet is not None:
            assert nalphabet == N.shape[0]
        assert context_size == 1
        self.nalphabet = N.shape[0]
        self.N = N.clone().detach()
        N = N.to(dtype=torch.float32) + prior
        self.p = torch.nan_to_num((N / N.sum(1, keepdim=True)).log(), torch.nan, neginf=-1e6)

    def context_size(self):
        return 1

    def forward(self, idx: torch.Tensor):
        return self.p[idx]


class OneLayerBigramModel(nn.Module):
    def __init__(self, *, nalphabet: int, context_size=1):
        super().__init__()
        self.nalphabet = nalphabet
        assert context_size == 1
        self.w = nn.Parameter(torch.ones((nalphabet,nalphabet)) / nalphabet)

    def context_size(self):
        return 1
    
    def forward(self, idx: torch.Tensor):
        return self.w[idx]


class EmbeddingMLP(nn.Module):
    def __init__(self, *, nalphabet: int, context_size: int, nembedding: int, nhidden: int):
        super().__init__()

        self.nalphabet = nalphabet
        self.nembedding = nembedding
        self._context_size = context_size
        self.nhidden = nhidden

        self.emb = nn.Embedding(nalphabet, nembedding)
        self.linear1 = nn.Linear(context_size * nembedding, nhidden)
        self.linear2 = nn.Linear(nhidden, nalphabet)

    def context_size(self):
        return self._context_size

    def forward(self, x):
        b = x.shape[0]
        t = x.shape[1]

        x = torch.cat([torch.zeros((b, self._context_size-1), dtype=torch.int64, device=x.device), x], 1)
        f = self.emb(x.unfold(1, self._context_size, 1))
        h = F.tanh(self.linear1(f.flatten(2)))
        return self.linear2(h)

