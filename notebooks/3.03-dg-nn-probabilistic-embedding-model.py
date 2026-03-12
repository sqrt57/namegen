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
from torch.profiler import profile, ProfilerActivity, record_function

from namegen.dataset import uk_towns_and_counties, InfiniteDataLoader
from namegen.modeling.model import BigramsModel, OneLayerBigramModel, ProbabilisticEmbeddingModel
from namegen.modeling.train import train_bigram_model, Trainer, Hyper, Result
from namegen.modeling.predict import generate, calculate_loss


# %%
device = torch.accelerator.current_accelerator().type
# device = "cpu"
print(f"Using {device} device")

# %%
dataset = uk_towns_and_counties('../data')

# %%
optimizer = torch.optim.AdamW
seed = 489044167
hyper = [
    Hyper(name='baseline', context_size=1, batch_size=100, nsteps=20000, lr=1e-3, optimizer=optimizer, seed=seed,
         model=OneLayerBigramModel),
    Hyper(name='embedding 3 2 50', context_size=3, batch_size=100, nsteps=20000, lr=1e-3, optimizer=optimizer, seed=seed,
         model=ProbabilisticEmbeddingModel, model_kwargs={ 'nembedding': 2, 'nhidden': 50 }),
    Hyper(name='embedding 5 4 100', context_size=5, batch_size=100, nsteps=20000, lr=1e-3, optimizer=optimizer, seed=seed,
         model=ProbabilisticEmbeddingModel, model_kwargs={ 'nembedding': 4, 'nhidden': 100 }),
    Hyper(name='embedding 8 8 200', context_size=8, batch_size=100, nsteps=40000, lr=1e-3, optimizer=optimizer, seed=seed,
         model=ProbabilisticEmbeddingModel, model_kwargs={ 'nembedding': 8, 'nhidden': 200 }),
]

# %%
trainer = Trainer(dataset, device=device)

# %%
results = []
for h in hyper:
    results.append(trainer.run_scenario(h))

# %%
for r in results:
    npars = sum(p.numel() for p in r.model.parameters())
    print(f"model={r.hyper.model.__name__} {npars=}")

# %%
fig, ax1 = plt.subplots(1, 1, figsize=(5,  4))

for result in results:
    ax1.plot(result.steps, result.train_metrics['loss'], label=result.hyper.name)
ax1.set_xlabel("Steps")
ax1.set_title("Train accuracy")
ax1.grid(True)
ax1.legend()

plt.plot()

# %%
for r in results:
    print(calculate_loss(dataset, r.model))

# %%
seed = 649431853

df = pd.DataFrame()

for r in results:
    torch.random.manual_seed(seed)
    df[r.hyper.name] = generate(dataset, r.model)

df

# %%
model = results[-1].model

df = pd.DataFrame()
for T in [0.1, 0.5, 0.9, 1.0, 1.1, 1.2, 1.5]:
    name = f"{T:.2}"
    torch.random.manual_seed(seed)
    df[name] = generate(dataset, model, T=T)

df
