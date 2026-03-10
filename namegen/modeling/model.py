import torch
from torch import nn
import torch

from namegen.dataset import Dataset

class BigramsModel(nn.Module):
    def __init__(self, N: torch.Tensor, prior: int | float = 0):
        super().__init__()
        self.N = N.clone().detach()
        N = N.to(dtype=torch.float32) + prior
        self.p = (N / N.sum(1, keepdim=True)).log()

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.squeeze()
        return self.p[x]

def train_bigram_model(dataset: Dataset, prior : int | float = 0):
    features, labels = dataset.get_features_and_labels(context_size=1)
    features = features.squeeze()
    N = torch.zeros((dataset.nalphabet, dataset.nalphabet), dtype=torch.int64)
    for i in range(features.shape[0]):
        N[features[i], labels[i]] += 1
    model = BigramsModel(N, prior)
    return model
