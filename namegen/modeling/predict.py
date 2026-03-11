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

    def generate(self, *, T=1, seed=None):
        self.model.eval()
        context_size: int = self.model.context_size
        if (seed is not None):
            torch.manual_seed(seed)
        current = [self.ctoi['_']] * context_size 
        result = []
        while True:
            in_p = torch.zeros((1, context_size, self.nalphabet))
            for i in range(context_size):
                in_p[0, i, current[i]] = 1.
            probabilities = F.softmax(self.model(in_p) / T, 1)[0]
            next = torch.multinomial(probabilities, 1).item()
            current = current[1:] + [next]
            next_char = self.alphabet[next]
            if next_char == '_':
                return ''.join(result)
            result.append(next_char)

def calculate_loss(dataset: Dataset, model: nn.Module):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    features, labels = dataset.get_features_and_labels(model.context_size)
    features = F.one_hot(features, num_classes=dataset.nalphabet).to(dtype=torch.float32)
    pred = model(features)
    return loss_fn(pred, labels)
