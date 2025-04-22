from .lmi import LMI as LMIBase
from .helpers import extract_model_config

import torch

class LMI:
    def __init__(self, model, *args, **kwargs):
        model_config = extract_model_config(model)
        self._inner = LMIBase(model_config, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def build(self, X: torch.Tensor,  epochs: int, lr: float):
        y = self._run_kmeans(X)
        self._train_model(X, y, epochs, lr)
        self._create_buckets(X)

