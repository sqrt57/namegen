from collections import Counter
from pathlib import Path

import pandas as pd
import torch

def read_uk_towns(data_path):
    return pd.read_csv(Path(data_path) / 'raw' / 'uk-towns.txt', skipinitialspace=True)

def uk_towns_and_counties_list(data_path: str) -> list[str]:
    df = read_uk_towns(data_path)
    return pd.concat([df['Town'], df['County']]).str.lower().drop_duplicates().sort_values().to_list()

def build_dataset(strings: list[str], context_size: int) -> list[tuple[str, str]]:
    dataset = []
    prefix = '_' * context_size
    postfix = '_'
    for s in strings:
        s = prefix + s + postfix
        for i in range(0, len(s)-context_size):
            context = s[i:i+context_size]
            target = s[i+context_size]
            dataset.append((context, target))
    return dataset

def get_char_counts(strings: list[str]) -> Counter:
    return Counter(''.join(strings))

def get_alphabet(strings: list[str]) -> str:
    counts = get_char_counts(strings)
    alphabet = "_"
    for char, count in counts.most_common():
        alphabet += char
    return alphabet

def get_char_to_index(alphabet: str) -> dict[str, int]:
    return {char: i for i, char in enumerate(alphabet)}

def get_features_and_labels(alphabet, dataset: list[tuple[str, str]]):
    ctoi = get_char_to_index(alphabet)
    X = []
    Y = []
    for context, target in dataset:
        x = [ctoi[c] for c in context]
        y = ctoi[target]
        X.append(x)
        Y.append(y)
    return torch.tensor(X), torch.tensor(Y)