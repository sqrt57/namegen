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


class RNN(nn.Module):
    def __init__(self, *, nalphabet: int, context_size: int, nembedding: int, nstate: int):
        super().__init__()

        self._nalphabet = nalphabet
        self._nembedding = nembedding
        self._nstate = nstate
        self._context_size = context_size

        self.emb = nn.Embedding(nalphabet, nembedding)
        self.input = nn.Linear(nembedding + nstate, nstate)
        self.output = nn.Linear(nstate, nalphabet)

    def context_size(self):
        return self._context_size

    def forward(self, w):
        b = w.shape[0]
        t = w.shape[1]
        assert w.shape == (b, t)
        state = torch.zeros(b, self._nstate, device=w.device)

        f = self.emb(w)
        assert f.shape == (b, t, self._nembedding)

        states = []
        for i in range(t):
            x = torch.cat((f[:,i,:], state), dim=1)
            assert x.shape == (b, self._nembedding + self._nstate)

            state_next = F.tanh(self.input(x))
            assert state_next.shape == (b, self._nstate)

            states.append(state_next)
            state = state_next

        states = torch.stack(states, dim=1)        
        assert states.shape == (b, t, self._nstate)

        y = self.output(states)
        assert y.shape == (b, t, self._nalphabet)
        return y

class LSTM(nn.Module):
    def __init__(self, *, nalphabet: int, context_size: int, nembedding: int, nstate: int):
        super().__init__()

        self._nalphabet = nalphabet
        self._nembedding = nembedding
        self._nstate = nstate
        self._context_size = context_size

        self.emb = nn.Embedding(self._nalphabet, self._nembedding)
        self.forget_input = nn.Linear(self._nembedding + 2*self._nstate, 2*self._nstate)
        self.hidden = nn.Linear(self._nembedding + 2*self._nstate, self._nstate)
        self.cell_update = nn.Linear(self._nembedding + self._nstate, self._nstate)
        self.output = nn.Linear(self._nstate, self._nalphabet)


    def context_size(self):
        return self._context_size

    def forward(self, x):
        b = x.shape[0]
        t = x.shape[1]
        assert x.shape == (b, t)
        cell = torch.zeros(b, self._nstate, device=x.device)
        hidden = torch.zeros(b, self._nstate, device=x.device)

        f = self.emb(x)
        assert f.shape == (b, t, self._nembedding)

        states = []
        for i in range(t):
            x_slice = f[:,i,:]
            assert x_slice.shape == (b, self._nembedding)

            gate_args = torch.cat((x_slice, hidden, cell), dim=1)
            assert gate_args.shape == (b, self._nembedding + 2*self._nstate)

            forget_input_gate = F.sigmoid(self.forget_input(gate_args))
            assert forget_input_gate.shape == (b, 2*self._nstate)

            cell_update_args = torch.cat((x_slice, hidden), dim=1)
            assert cell_update_args.shape == (b, self._nembedding + self._nstate)

            cell = forget_input_gate[:,:self._nstate] * cell + forget_input_gate[:,self._nstate:] * F.tanh(self.cell_update(cell_update_args))
            assert cell.shape == (b, self._nstate)

            new_gate_args = torch.cat((x_slice, hidden, cell), dim=1)
            assert new_gate_args.shape == (b, self._nembedding + 2*self._nstate)

            hidden_gate = self.hidden(new_gate_args)
            assert hidden_gate.shape == (b, self._nstate)
            
            hidden = hidden_gate * F.tanh(cell)
            assert hidden.shape == (b, self._nstate)

            states.append(hidden)

        states = torch.stack(states, dim=1)        
        assert states.shape == (b, t, self._nstate)

        y = self.output(states)
        assert y.shape == (b, t, self._nalphabet)

        return y