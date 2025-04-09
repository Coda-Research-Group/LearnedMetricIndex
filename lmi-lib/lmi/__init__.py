from .lmi import LMI as LMIBase

import torch

class LMI:
    def __init__(self, *args, **kwargs):
        self._inner = LMIBase(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def build(self, X: torch.Tensor,  epochs: int, lr: float):
        y = self._run_kmeans(X)
        self._train_model(X, y, epochs, lr)
        self._create_buckets(X)

