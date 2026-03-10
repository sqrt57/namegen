from collections import Counter, namedtuple
from functools import cache
from pathlib import Path

import pandas as pd
import torch

__names__ = [
    'Dataset',
    'uk_towns_and_counties',
    'empty_dataset',
]

def read_uk_towns(data_path):
    return pd.read_csv(Path(data_path) / 'raw' / 'uk-towns.txt', skipinitialspace=True)

def uk_towns_and_counties_list(data_path: str) -> list[str]:
    df = read_uk_towns(data_path)
    return pd.concat([df['Town'], df['County']]).str.lower().drop_duplicates().sort_values().to_list()

def get_char_counts(strings: list[str]) -> Counter:
    return Counter(''.join(strings))

def get_alphabet(strings: list[str]) -> str:
    counts = get_char_counts(strings)
    alphabet = ""
    for char, count in counts.most_common():
        alphabet += char
    return alphabet

def get_char_to_index(alphabet: str) -> dict[str, int]:
    return {char: i for i, char in enumerate(alphabet)}


Batch = namedtuple("Batch", "features, labels")
    
class Dataset:
    def __init__(self, strings: list[str], alphabet: str | None = None):
        self.strings = list(strings)
        self.alphabet = get_alphabet(strings)
        if alphabet is not None:
            self.alphabet = alphabet + ''.join([c for c in self.alphabet if c not in alphabet])
        if '_' not in self.alphabet:
            self.alphabet = '_' + self.alphabet
        self.nalphabet = len(self.alphabet)
        self.ctoi = get_char_to_index(self.alphabet)
    
    def build_dataset(self, context_size: int) -> list[tuple[str, str]]:
        dataset = []
        prefix = '_' * context_size
        postfix = '_'
        for s in self.strings:
            s = prefix + s + postfix
            for i in range(0, len(s)-context_size):
                context = s[i:i+context_size]
                target = s[i+context_size]
                dataset.append((context, target))
        return dataset
    
    def get_features_and_labels(self, context_size: int):
        features = []
        labels = []
        prefix = torch.ones(context_size, dtype=torch.int) * self.ctoi['_']
        postfix = torch.tensor([self.ctoi['_']])
        for s in self.strings:
            line = torch.tensor([self.ctoi[c] for c in s])
            features_batch = torch.cat([prefix, line]).unfold(0, context_size, 1)
            labels_batch = torch.cat([line, postfix])
            features.append(features_batch)
            labels.append(labels_batch)
        return Batch(features=torch.cat(features), labels=torch.cat(labels))
    
    def get_features_and_labels_batches(self, context_size: int, batch_size, shuffle=False):
        features, labels = self.get_features_and_labels(context_size)
        nitems = features.shape[0]

        if shuffle:
            indices = torch.randperm(nitems)
        else:
            indices = torch.arange(nitems)

        for i in range(0, nitems, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield Batch(features=features[batch_indices], labels=labels[batch_indices])


@cache
def uk_towns_and_counties(data_path: str) -> Dataset:
    df = read_uk_towns(data_path)
    towns_and_cities = pd.concat([df['Town'], df['County']]).str.lower().drop_duplicates().sort_values()
    return Dataset(towns_and_cities.to_list())

def empty_dataset(alphabet: str | None = None) -> Dataset:
    return Dataset([], alphabet=alphabet)
