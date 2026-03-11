import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    'BigramsModel',
    'OneLayerBigramModel',
]

class BigramsModel(nn.Module):
    def __init__(self, N: torch.Tensor, prior: int | float = 0, *, nalphabet, context_size=1):
        super().__init__()
        assert context_size == 1
        assert nalphabet == N.shape[0]
        self.nalphabet = nalphabet
        self.context_size = context_size
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
        self.context_size = context_size

        self.w = nn.Parameter(torch.ones((nalphabet,nalphabet)) / nalphabet)
    
    def forward(self, x: torch.Tensor):
        return x.reshape(-1, self.nalphabet) @ self.w


class ProbabilisticEmbeddingModel(nn.Module):
    def __init__(self, *, nalphabet: int, context_size: int, nembedding: int, nhidden: int):
        super().__init__()

        self.nalphabet = nalphabet
        self.nembedding = nembedding
        self.context_size = context_size
        self.nhidden = nhidden

        self.emb = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros((nembedding, nalphabet)), mode='fan_in', nonlinearity='linear'))

        self.H = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros((nhidden, context_size * nembedding)), mode='fan_in', nonlinearity='tanh'))
        self.d = nn.Parameter(torch.zeros(nhidden))

        self.U = nn.Parameter(nn.init.kaiming_uniform_(torch.zeros((nalphabet, nhidden)), mode='fan_in', nonlinearity='linear'))
        self.b = nn.Parameter(torch.zeros(nalphabet))


    def forward(self, x):
        d = x.shape[0]
        assert x.shape == (d, self.context_size, self.nalphabet)

        f = (x @ self.emb.T).reshape(-1, self.context_size * self.nembedding)

        h = torch.tanh(f @ self.H.T + self.d.reshape(1, -1))

        return h @ self.U.T + self.b.reshape(1, -1)
