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
alphabet = dataset.get_alphabet(data)
nalphabet = len(alphabet)
ctoi = dataset.get_char_to_index(alphabet)
print(nalphabet)
print(alphabet)
print(ctoi)

# %%
N = torch.ones((nalphabet, nalphabet), dtype=torch.float64) * 0.1
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
p.dtype


# %%
def generate():
    current = ctoi['_']
    result = ""
    while True:
        current = torch.multinomial(p[current], 1).item()
        if alphabet[current] == '_':
            return result;
        result += alphabet[current]


# %%
torch.random.manual_seed(998190804)
[generate() for i in range(20)]
