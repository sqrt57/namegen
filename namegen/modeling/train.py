
import collections
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


from namegen.dataset import Dataset, InfiniteDataLoader
from namegen.modeling.model import BigramsModel

__all__ = [
    'train_bigram_model',
    'Hyper',
    'Result',
    'Trainer',
]


def train_bigram_model(dataset: Dataset, prior : int | float = 0):
    features = dataset.features
    labels = dataset.labels
    N = torch.zeros((dataset.nalphabet, dataset.nalphabet), dtype=torch.int64)
    for i in range(features.shape[0]):
        for j in range(features.shape[1]):
            if labels[i, j] >= 0:
                N[features[i, j], labels[i, j]] += 1
    model = BigramsModel(N, prior, nalphabet=dataset.nalphabet)
    return model

Hyper = collections.namedtuple('Hyper', 'name context_size model batch_size nsteps model_kwargs optimizer lr optimizer_kwargs shuffle seed',
                               defaults=({}, torch.optim.Adam, 3e-4, {}, True, None))
Result = collections.namedtuple('Result', 'hyper model steps train_metrics')

class Trainer:
    def __init__(self, dataset: Dataset, device="cpu"):
        self.dataset = dataset
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = device

    def run_scenario(self, hyper: Hyper):
        print(f"Running scenario {hyper.name}: model={hyper.model.__name__} seed={hyper.seed} "
              + f"context_size={hyper.context_size} batch_size={hyper.batch_size} lr={hyper.lr}")
        if (hyper.seed is None):
            torch.seed()
        else:
            torch.manual_seed(hyper.seed)
        model = hyper.model(nalphabet=self.dataset.nalphabet, context_size=hyper.context_size, **hyper.model_kwargs).to(device=self.device)
        optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr, **hyper.optimizer_kwargs)
        dataloader = InfiniteDataLoader(self.dataset, hyper.batch_size, self.device)
        features = self.dataset.features.to(device=self.device)
        labels = self.dataset.labels.to(device=self.device)

        steps = []
        train_metrics = { 'loss': [] }
        model.train()
        
        for nstep in tqdm(range(hyper.nsteps), desc="Steps"):
            features, labels = dataloader.next()
            model.train()
            pred = model(features)
            loss = self.loss_fn(pred.flatten(0, 1), labels.flatten(0, 1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if nstep % 10 == 0:
                model.eval()
                pred = model.forward(features)
                loss = self.loss_fn(pred.flatten(0, 1), labels.flatten(0, 1))
                model.train()
                steps.append(nstep)
                train_metrics['loss'].append(loss.item())

        return Result(hyper=hyper, model=model.to("cpu"), steps=steps, train_metrics=train_metrics)
