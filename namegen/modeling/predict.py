import torch
from torch import nn
import torch.nn.functional as F

from namegen.dataset import Dataset

def generate(dataset: Dataset, model: nn.Module, N=20, T=1, max_len=100):
    model.eval()
    context_size: int = model.context_size()
    x = torch.zeros(N, 1, dtype=torch.int64)
    for i in range(max_len):
        x_win = x[:, -context_size:] if x.shape[1] >= context_size else x
        logits = model(x_win)
        logits = logits[:, -1, :]
        probabilities = F.softmax(logits / T, dim=1)
        next = torch.multinomial(probabilities, 1)
        x = torch.cat((x, next), dim=1)
        if (x[:,1:] == 0).any(dim=1).all(dim=0).item():
            break
    result = []
    for n in range(N):
        row = x[n, 1:].tolist()
        if 0 in row:
            row = row[:row.index(0)]
        word = ''.join([dataset.itoc[i] for i in row])
        result.append(word)
    return result


def calculate_loss(dataset: Dataset, model: nn.Module):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    pred = model(dataset.features)
    return loss_fn(pred.flatten(0, 1), dataset.labels.flatten(0, 1))
