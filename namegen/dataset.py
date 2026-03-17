from collections import Counter, namedtuple
from functools import cache
from pathlib import Path

import pandas as pd
import torch

__all__ = [
    'Dataset',
    'uk_towns_and_counties',
    'empty_dataset',
]

def read_uk_towns(data_path):
    return pd.read_csv(Path(data_path) / 'raw' / 'uk-towns.csv', skipinitialspace=True)

def uk_towns_and_counties_list(data_path: str) -> list[str]:
    df = read_uk_towns(data_path)
    lst = pd.concat([df['Town'], df['County']]).str.lower().drop_duplicates().to_list()
    for sep in ['/', '(', ')']:
        result = []
        for s in lst:
            result.extend(s.split(sep))
        lst = result
    return sorted(list(set([s.strip() for s in result if s.strip() != ''])))

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


Batch = namedtuple("Batch", "features labels")
    
class Dataset:
    def __init__(self, strings: list[str], alphabet: str | None = None):
        self.strings = list(strings)
        self.alphabet = get_alphabet(strings)
        self.max_word_length = max(len(s) for s in strings)
        if alphabet is not None:
            self.alphabet = alphabet + ''.join([c for c in self.alphabet if c not in alphabet])
        if '_' not in self.alphabet:
            self.alphabet = '_' + self.alphabet
        self.nalphabet = len(self.alphabet)
        self.ctoi = {char: i for i, char in enumerate(self.alphabet)}
        self.itoc = {i: char for i, char in enumerate(self.alphabet)}
        self.features, self.labels = self.get_features_and_labels()
    
    def get_features_and_labels(self):
        features = []
        labels = []
        for s in self.strings:
            line = torch.tensor([self.ctoi[c] for c in s], dtype=torch.int64)
            features_line = torch.zeros(self.max_word_length + 1, dtype=torch.int64)
            labels_line = torch.zeros(self.max_word_length + 1, dtype=torch.int64)
            features_line[1:1+len(line)] = line
            labels_line[:len(line)] = line
            labels_line[len(line)+1:] = -1
            features.append(features_line)
            labels.append(labels_line)
        return torch.stack(features), torch.stack(labels)


class InfiniteDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, device='cpu'):
        self.features = dataset.features.to(device=device)
        self.labels = dataset.labels.to(device=device)
        self.batch_size = batch_size
        self.ones = torch.ones(dataset.features.shape[0], device=device)

    def next(self):
        ix = torch.multinomial(self.ones, self.batch_size)
        return self.features[ix], self.labels[ix]


@cache
def uk_towns_and_counties(data_path: str) -> Dataset:
    towns_and_counties = uk_towns_and_counties_list(data_path)
    return Dataset(towns_and_counties)

def empty_dataset(alphabet: str | None = None) -> Dataset:
    return Dataset([], alphabet=alphabet)
