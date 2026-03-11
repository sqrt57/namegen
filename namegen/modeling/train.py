
import collections
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm


from namegen.dataset import Dataset, get_features_and_labels_batches
from namegen.modeling.model import BigramsModel

__all__ = [
    'train_bigram_model',
    'Hyper',
    'Result',
    'Metrics',
    'LossMetrics',
    'AccuracyMetrics',
    'Trainer',
]


def train_bigram_model(dataset: Dataset, prior : int | float = 0):
    features, labels = dataset.get_features_and_labels(context_size=1)
    features = features.squeeze()
    N = torch.zeros((dataset.nalphabet, dataset.nalphabet), dtype=torch.int64)
    for i in range(features.shape[0]):
        N[features[i], labels[i]] += 1
    model = BigramsModel(N, prior, nalphabet=dataset.nalphabet)
    return model

Hyper = collections.namedtuple('Hyper', 'name dataset context_size model batch_size nepochs model_kwargs optimizer lr optimizer_kwargs shuffle seed device',
                               defaults=({}, torch.optim.Adam, 3e-4, {}, True, None, "cpu"))
Result = collections.namedtuple('Result', 'hyper model epochs train_metrics')


class Metrics:
    def name(self) -> str:
        raise NotImplementedError()

    def batch(self, model, features_batch, labels_batch, prediction, loss) -> tuple:
        raise NotImplementedError()

    def summarize(self, chunks) -> Any:
        raise NotImplementedError()


class LossMetrics(Metrics):
    def name(self):
        return "Loss"

    def batch(self, model, features_batch, labels_batch, prediction, loss):
        return (loss, features_batch.shape[0])
    
    def summarize(self, chunks):
        total_loss = sum(chunk[0] for chunk in chunks)
        total_count = sum(chunk[1] for chunk in chunks)
        return total_loss / total_count

   
class AccuracyMetrics(Metrics):
    def name(self):
        return "Accuracy"

    def batch(self, model, features_batch, labels_batch, prediction, loss):
        correct = (prediction.argmax(1) == labels_batch).sum().item()
        return (correct, features_batch.shape[0])
    
    def summarize(self, chunks):
        total_correct = sum(chunk[0] for chunk in chunks)
        total_count = sum(chunk[1] for chunk in chunks)
        return total_correct * 100 / total_count


class Trainer:
    def __init__(self, metrics: list[Metrics] | None = None):
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.metrics = metrics or [LossMetrics(), AccuracyMetrics()]

    def run_scenario(self, hyper: Hyper):
        print(f"Running scenario {hyper.name}: model={hyper.model.__name__} seed={hyper.seed} "
              + f"context_size={hyper.context_size} batch_size={hyper.batch_size} lr={hyper.lr}")
        if (hyper.seed is None):
            torch.seed()
        else:
            torch.manual_seed(hyper.seed)
        model = hyper.model(nalphabet=hyper.dataset.nalphabet, context_size=hyper.context_size, **hyper.model_kwargs).to(hyper.device)
        optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr, **hyper.optimizer_kwargs)
        total_chunk = hyper.dataset.get_features_and_labels(hyper.context_size, device=hyper.device)

        epochs = []
        train_metrics = { metric.name(): [] for metric in self.metrics }

        def batch(features_batch, labels_batch):
            model.train()
            optimizer.zero_grad()
            features_batch_one_hot = F.one_hot(features_batch, num_classes=hyper.dataset.nalphabet).to(dtype=torch.float32)
            pred = model.forward(features_batch_one_hot)
            loss = self.loss_fn(pred, labels_batch)
            (loss / features_batch.shape[0]).backward()
            optimizer.step()
            return pred, loss

        def testbatch(model, features_batch, labels_batch):
            with torch.no_grad():
                model.eval()
                features_batch_one_hot = F.one_hot(features_batch, num_classes=hyper.dataset.nalphabet).to(dtype=torch.float32)
                pred = model.forward(features_batch_one_hot)
                loss = self.loss_fn(pred, labels_batch)
            return pred, loss

        def run_epoch(epoch, train):
            epochs.append(epoch)

            if (train):
                train_metric_chunks = { metric.name(): [] for metric in self.metrics }
                for chunk in get_features_and_labels_batches(total_chunk, batch_size=hyper.batch_size, shuffle=hyper.shuffle, device=hyper.device):
                    pred, loss = batch(chunk.features, chunk.labels)
                    for metric in self.metrics:
                        train_metric_chunks[metric.name()].append(metric.batch(model, chunk.features, chunk.labels, pred, loss.item()))
                for metric in self.metrics:
                    train_metrics[metric.name()].append(metric.summarize(train_metric_chunks[metric.name()]))
            else:
                pred, loss = testbatch(model, total_chunk.features, total_chunk.labels)
                for metric in self.metrics:
                    train_metrics[metric.name()].append(metric.summarize([metric.batch(model, total_chunk.features, total_chunk.labels, pred, loss.item())]))
        
        run_epoch(0, train=False)

        for epoch in tqdm(range(hyper.nepochs), desc="Epochs"):
            run_epoch(epoch + 1, train=True)

        return Result(hyper=hyper, model=model.to("cpu"), epochs=epochs, train_metrics=train_metrics)
