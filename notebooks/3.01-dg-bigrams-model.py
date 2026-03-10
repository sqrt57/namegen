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
import torch.nn.functional as F

from namegen.dataset import uk_towns_and_counties
from namegen.modeling.model import BigramsModel, train_bigram_model
from namegen.modeling.predict import Generator, calculate_loss


# %%
dataset = uk_towns_and_counties('../data')

# %%
features, labels = dataset.get_features_and_labels(1)
print(features.shape)
print(torch.unique(features))
print(labels.shape)
print(torch.unique(labels))

# %%
print(dataset.nalphabet)
print(dataset.alphabet)
print(dataset.ctoi)

# %%
random_model = BigramsModel(torch.ones((dataset.nalphabet, dataset.nalphabet)))
random_generator = Generator(dataset, random_model)

model_00 = train_bigram_model(dataset, 0)
model_05= train_bigram_model(dataset, 0.5)
model_10 = train_bigram_model(dataset, 1)
model = model_05
generator = Generator(dataset, model)

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
torch.random.manual_seed(998190804)
[generator.generate() for i in range(20)]

# %%
torch.random.manual_seed(998190804)
[random_generator.generate() for i in range(20)]

# %%
print(model(features).shape)
print(calculate_loss(dataset, model_00))
print(calculate_loss(dataset, model_05))
print(calculate_loss(dataset, model_10))
print(calculate_loss(dataset, random_model))
