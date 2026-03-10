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
from namegen.modeling.model import BigramsModel, OneLayerBigramModel
from namegen.modeling.train import train_bigram_model, Trainer, Hyper, Result
from namegen.modeling.predict import Generator, calculate_loss


# %%
dataset = uk_towns_and_counties('../data')

# %%
baseline_model = train_bigram_model(dataset, 0.1)
random_model = BigramsModel(torch.ones((dataset.nalphabet, dataset.nalphabet)))

# %%
hyper = [
    Hyper(name='3 epochs', dataset=dataset, context_size=1, model=OneLayerBigramModel, batch_size=100, nepochs=3, lr=1e-3),
    Hyper(name='10 epochs', dataset=dataset, context_size=1, model=OneLayerBigramModel, batch_size=100, nepochs=10, lr=1e-3),
    Hyper(name='30 epochs', dataset=dataset, context_size=1, model=OneLayerBigramModel, batch_size=100, nepochs=30, lr=1e-3),
    Hyper(name='100 epochs', dataset=dataset, context_size=1, model=OneLayerBigramModel, batch_size=100, nepochs=100, lr=1e-3),
]

# %%
trainer = Trainer()

# %%
results = []
for h in hyper:
    results.append(trainer.run_scenario(h))

# %%
for r in results:
    npars = sum(p.numel() for p in r.model.parameters())
    print(f"model={r.hyper.model.__name__} {npars=}")

# %%
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,  8))

for result in results:
    ax1.plot(result.epochs, result.train_metrics['Accuracy'], label=result.hyper.name)
# ax1.set_ylim(80., 100.)
ax1.set_ylim(0., 100.)
ax1.set_xlabel("Epoch")
ax1.set_title("Train accuracy")
ax1.grid(True)
ax1.legend()

# for result in results:
#     ax2.plot(result.epochs, result.validation_metrics['Accuracy'], label=result.hyper.name)
# # ax2.set_ylim(80., 100.)
# ax2.set_ylim(0., 100.)
# ax2.set_xlabel("Epoch")
# ax2.set_title("Test accuracy")
# ax2.grid(True)
# ax2.legend()

for result in results:
    ax3.plot(result.epochs, result.train_metrics['Loss'], label=result.hyper.name)
# ax3.set_ylim(0., 5.)
ax3.set_xlabel("Epoch")
ax3.set_title("Train loss")
ax3.grid(True)
ax3.legend()

# for result in results:
#     ax4.plot(result.epochs, result.validation_metrics['Loss'], label=result.hyper.name)
# ax4.set_ylim(0., 5.)
# ax4.set_xlabel("Epoch")
# ax4.set_title("Test loss")
# ax4.grid(True)
# ax4.legend()

fig.tight_layout()
plt.plot()

# %%
df = pd.DataFrame()
seeds = [
    656644231,	600012153,	909928419,	186362313,	579571659,
    571489154,	798841376,	649431853,	69803936,	351537320,
    438382231,	157521886,	785863303,	534713611,	536761809,
    754169915,	62491729,	951896890,	445648194,	804560408,
]

random_generator = Generator(dataset, random_model)
df['random'] = [random_generator.generate(seed=seed) for seed in seeds]

for r in results:
    generator = Generator(dataset, r.model)
    df[r.hyper.name] = [generator.generate(seed=seed) for seed in seeds]

baseline_generator = Generator(dataset, baseline_model)
df['baseline'] = [baseline_generator.generate(seed=seed) for seed in seeds]

df

# %%
print(calculate_loss(dataset, random_model))
for r in results:
    print(calculate_loss(dataset, r.model))
print(calculate_loss(dataset, baseline_model))

# %%
