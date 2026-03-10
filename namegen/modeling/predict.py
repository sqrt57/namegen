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

    def generate(self, start_symbol='_'):
        current = self.ctoi[start_symbol]
        result = []
        while True:
            probabilities = F.softmax(self.model(current))
            current = int(torch.multinomial(probabilities, 1).item())
            current_char = self.alphabet[current]
            if current_char == '_':
                return ''.join(result)
            result.append(current_char)

def calculate_loss(dataset: Dataset, model: nn.Module):
    loss_fn = nn.CrossEntropyLoss()
    features, labels = dataset.get_features_and_labels(1)
    return loss_fn(model(features), labels)