# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("..")

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

from namegen import dataset

# %%
data = dataset.uk_towns_and_counties_list('../data')

# %%
print(len(data))
print(data[:10])

# %%
ds = dataset.build_dataset(data, 1)
print(len(ds))
print(ds[:10])


# %%
(features, labels) = dataset.get_features_and_labels(dataset.get_alphabet(data), ds)
print(features.shape)
print(torch.unique(features))
print(labels.shape)
print(torch.unique(labels))

# %%
alphabet = dataset.get_alphabet(data)
nalphabet = len(alphabet)
ctoi = dataset.get_char_to_index(alphabet)
print(nalphabet)
print(alphabet)
print(ctoi)

# %%
N = torch.zeros((nalphabet, nalphabet), dtype=torch.int32)
for context, target in ds:
    N[ctoi[context], ctoi[target]] += 1

# %%
fig = plt.figure(figsize=(16,16))
ax = fig.add_axes([0,0,1,1])
ax.imshow(N, cmap='Blues')
for i in range(nalphabet):
    for j in range(nalphabet):
        chstr = alphabet[i] + ' ' + alphabet[j]
        ax.text(j, i, chstr, ha="center", va="bottom", color='gray', size='large')
        ax.text(j, i, N[i, j].item(), ha="center", va="top", color='gray', size='large')
ax.axis('off');

# %%
p = N / N.sum(1, keepdim=True)
print(p.dtype)
print(p.shape)


# %%
class BigramsModel(nn.Module):
    def __init__(self, N, prior=0):
        super().__init__()
        N = N.clone().detach().to(dtype=torch.float32)
        N += prior
        self.p = N / N.sum(1, keepdim=True)

    def forward(self, x):
        if torch.is_tensor(x):
            x = x.squeeze()
        return self.p[x]

class Generator:
    def __init__(self, alphabet, model):
        self.alphabet = alphabet
        self.nalphabet = len(alphabet)
        self.ctoi = {ch: i for i, ch in enumerate(alphabet)}
        self.model = model

    def generate(self, start_symbol='_'):
        current = self.ctoi[start_symbol]
        result = []
        while True:
            probabilities = self.model(current)
            current = torch.multinomial(probabilities, 1).item()
            current_char = self.alphabet[current]
            if current_char == '_':
                return ''.join(result)
            result.append(current_char)

random_model = BigramsModel(torch.ones((nalphabet,nalphabet)))
random_generator = Generator(alphabet, random_model)

model0 = BigramsModel(N)
model05 = BigramsModel(N, 0.5)
model1 = BigramsModel(N, 1)
model = model05
generator = Generator(alphabet, model)

# %%
torch.random.manual_seed(998190804)
[generator.generate() for i in range(20)]

# %%
torch.random.manual_seed(998190804)
[random_generator.generate() for i in range(20)]

# %%
print(features.shape)
print(labels.shape)
print(model(features).shape)
loss = nn.NLLLoss()
print(loss(torch.log(model0(features)), labels))
print(loss(torch.log(model05(features)), labels))
print(loss(torch.log(model1(features)), labels))
print(loss(torch.log(random_model(features)), labels))

# %%
loss = nn.CrossEntropyLoss()
print(loss(torch.log(model0(features)), labels))
print(loss(torch.log(model05(features)), labels))
print(loss(torch.log(model1(features)), labels))
print(loss(torch.log(random_model(features)), labels))
