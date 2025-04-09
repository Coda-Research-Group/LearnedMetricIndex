from .lmi import LMI as LMIBase

class LMI:
    def __init__(self, *args, **kwargs):
        self._inner = LMIBase(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def build(self, X, epochs, lr):
        y = self._run_kmeans(X)
        self._train_model(X, y, epochs, lr)
        self._create_buckets(X)

