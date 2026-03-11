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

from namegen.dataset import uk_towns_and_counties
from namegen.modeling.model import BigramsModel
from namegen.modeling.train import train_bigram_model
from namegen.modeling.predict import generate, calculate_loss


# %%
dataset = uk_towns_and_counties('../data')

# %%
print(dataset.features.shape)
print(torch.unique(dataset.features))
print(dataset.labels.shape)
print(torch.unique(dataset.labels))

# %%
print(dataset.nalphabet)
print(dataset.alphabet)
print(dataset.max_word_length)
print(dataset.ctoi)
print(dataset.itoc)

# %%
random_model = BigramsModel(torch.ones((dataset.nalphabet, dataset.nalphabet)))

model_00 = train_bigram_model(dataset, 0)
model_05 = train_bigram_model(dataset, 0.5)
model_10 = train_bigram_model(dataset, 1)
model = model_05

# %%
fig = plt.figure(figsize=(16,16))
ax = fig.add_axes([0,0,1,1])
ax.imshow(model_00.N, cmap='Blues')
for i in range(dataset.nalphabet):
    for j in range(dataset.nalphabet):
        chstr = dataset.alphabet[i] + ' ' + dataset.alphabet[j]
        ax.text(j, i, chstr, ha="center", va="bottom", color='gray', size='large')
        ax.text(j, i, model_00.N[i, j].item(), ha="center", va="top", color='gray', size='large')
ax.axis('off');

# %%
print(calculate_loss(dataset, model_00))
print(calculate_loss(dataset, model_05))
print(calculate_loss(dataset, model_10))
print(calculate_loss(dataset, random_model))

# %%
torch.random.manual_seed(998190804)
generate(dataset, model)

# %%
df = pd.DataFrame()

torch.random.manual_seed(998190804)
df['random'] = generate(dataset, random_model)

torch.random.manual_seed(998190804)
df['model_00'] = generate(dataset, model_00)

torch.random.manual_seed(998190804)
df['model_05'] = generate(dataset, model_05)

torch.random.manual_seed(998190804)
df['model_10'] = generate(dataset, model_10)

df

# %%
df = pd.DataFrame()

torch.random.manual_seed(998190804)
df['0.1'] = generate(dataset, model, T=0.1)

torch.random.manual_seed(998190804)
df['0.9'] = generate(dataset, model, T=0.9)

torch.random.manual_seed(998190804)
df['1.0'] = generate(dataset, model, T=1.0)

torch.random.manual_seed(998190804)
df['1.1'] = generate(dataset, model, T=1.1)

torch.random.manual_seed(998190804)
df['1.5'] = generate(dataset, model, T=1.5)

df

# %%
