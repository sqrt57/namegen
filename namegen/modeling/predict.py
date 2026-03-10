import torch
from torch import nn
import torch.nn.functional as F

from namegen.dataset import Dataset

class Generator:
    def __init__(self, dataset: Dataset, model: nn.Module):
        self.dataset = dataset
        self.alphabet = dataset.alphabet
        self.nalphabet = len(self.alphabet)
        self.ctoi = {ch: i for i, ch in enumerate(self.alphabet)}
        self.model = model

    def generate(self, *, start_symbol='_', T=1, seed=None):
        if (seed is not None):
            torch.manual_seed(seed)
        current = self.ctoi[start_symbol]
        result = []
        while True:
            in_p = torch.zeros((1, self.nalphabet))
            in_p[0, current] = 1.
            probabilities = F.softmax(self.model(in_p) / T, 1)[0]
            current = int(torch.multinomial(probabilities, 1).item())
            current_char = self.alphabet[current]
            if current_char == '_':
                return ''.join(result)
            result.append(current_char)

def calculate_loss(dataset: Dataset, model: nn.Module):
    loss_fn = nn.CrossEntropyLoss()
    features, labels = dataset.get_features_and_labels(1)
    features = F.one_hot(features, num_classes=dataset.nalphabet).to(dtype=torch.float32)
    pred = model(features)
    return loss_fn(pred, labels)
