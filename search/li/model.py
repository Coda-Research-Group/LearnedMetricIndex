from dataclasses import astuple, dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
import torch.utils.data
from chromadb.li_index.search.li.clustering import ClusteringAlgorithm
from chromadb.li_index.search.li.Logger import Logger
from torch import nn
from torch.nn import Linear, ReLU, Sequential

torch.manual_seed(2023)
np.random.seed(2023)


@dataclass(frozen=True)
class ModelParameters:
    clustering_algorithm: ClusteringAlgorithm
    model_type: str
    epochs: int
    lr: float
    n_categories: int

    def __iter__(self):
        return iter(astuple(self))


supported_models: Dict[str, Callable[[int, int], Sequential]] = {
    "MLP": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 128),
        ReLU(),
        Linear(128, output_dim),
    ),
    "MLP-2": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 64),
        ReLU(),
        Linear(64, output_dim),
    ),
    "MLP-3": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 256),
        ReLU(),
        Linear(256, output_dim),
    ),
    "MLP-4": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 512),
        ReLU(),
        Linear(512, output_dim),
    ),
    "MLP-5": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 256),
        ReLU(),
        Linear(256, 128),
        ReLU(),
        Linear(128, output_dim),
    ),
    "MLP-6": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 32),
        ReLU(),
        Linear(32, output_dim),
    ),
    "MLP-7": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 16),
        ReLU(),
        Linear(16, output_dim),
    ),
    "MLP-8": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 8),
        ReLU(),
        Linear(8, output_dim),
    ),
    "MLP-9": lambda input_dim, output_dim: Sequential(
        Linear(input_dim, 8),
        ReLU(),
        Linear(8, 16),
        ReLU(),
        Linear(16, output_dim),
    ),
}


def init_layers(model_type: Optional[str], input_dim: int, output_dim: int):
    if model_type not in supported_models:
        raise ValueError(f"Model type {model_type} not supported.")

    return supported_models[model_type](input_dim, output_dim)


class Model(nn.Module):
    def __init__(
        self, input_dim=768, output_dim=1000, model_type: Optional[str] = None
    ):
        super().__init__()
        self.layers = init_layers(model_type, input_dim, output_dim)
        self.n_output_neurons = output_dim

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        outputs = self.layers(x)
        return outputs


def data_X_to_torch(data) -> torch.FloatTensor:
    """Creates torch training data."""
    data_X = torch.from_numpy(np.array(data).astype(np.float32))
    return data_X


def data_to_torch(data, labels) -> Tuple[torch.FloatTensor, torch.LongTensor]:
    """Creates torch training data and labels."""
    data_X = data_X_to_torch(data)
    data_y = torch.as_tensor(torch.from_numpy(labels), dtype=torch.long)
    return data_X, data_y


def get_device() -> torch.device:
    """Gets the `device` to be used by torch.
    This arugment is needed to operate with the PyTorch model instance.

    Returns
    ------
    torch.device
        Device
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    return device


class NeuralNetwork(Logger):
    """The neural network class corresponding to every inner node.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    loss : torch.nn, optional
        The loss function, the default is torch.nn.CrossEntropyLoss.
    lr : float, optional
        The learning rate, the default is 0.001.
    model_type : str, optional
        The model type, the default is 'MLP'.
    class_weight : torch.FloatTensor, optional
        The class weights, the default is None.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        loss=torch.nn.CrossEntropyLoss,
        lr=0.1,
        model_type="MLP",
        class_weight=None,
    ):
        self.device = get_device()
        self.model = Model(input_dim, output_dim, model_type=model_type).to(self.device)
        if not isinstance(class_weight, type(None)):
            self.loss = loss(weight=class_weight.to(self.device))
        else:
            self.loss = loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        data_X: torch.FloatTensor,
        data_y: torch.LongTensor,
        epochs=500,
        logger=None,
    ):
        step = epochs // 10
        losses = []
        if logger:
            logger.debug(f"Epochs: {epochs}, step: {step}")
        for ep in range(epochs):
            pred_y = self.model(data_X.to(self.device))
            curr_loss = self.loss(pred_y, data_y.to(self.device))
            if ep % step == 0 and ep != 0:
                if logger:
                    logger.debug(f"Epoch {ep} | Loss {curr_loss.item()}")
            losses.append(curr_loss.item())

            self.model.zero_grad()
            curr_loss.backward()

            self.optimizer.step()
        return losses

    def train_batch(self, dataset, epochs=5, logger=None):
        step = epochs // 10
        step = step if step > 0 else 1
        losses = []
        if logger:
            logger.debug(f"Epochs: {epochs}, step: {step}")
        for ep in range(epochs):
            for data_X, data_y in iter(dataset):
                pred_y = self.model(data_X.to(self.device))
                curr_loss = self.loss(pred_y, data_y.to(self.device))

            if ep % step == 0 and ep != 0:
                if logger:
                    logger.debug(f"Epoch {ep} | Loss {curr_loss.item():.5f}")
            losses.append(curr_loss.item())

            self.model.zero_grad()
            curr_loss.backward()

            self.optimizer.step()
        return losses

    def predict(self, data_X: torch.FloatTensor):
        """Collects predictions for multiple data points (used in structure building)."""
        self.model = self.model.to(self.device)
        self.model.eval()

        all_outputs = torch.tensor([], device=self.device)
        with torch.no_grad():
            outputs = self.model(data_X.to(self.device))
            all_outputs = torch.cat((all_outputs, outputs), 0)

        _, y_pred = torch.max(all_outputs, 1)
        return y_pred.cpu().numpy()

    def predict_proba(self, data_X: torch.FloatTensor):
        """Collects predictions for a single data point (used in query predictions)."""
        self.model = self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(data_X.to(self.device))

        if outputs.dim() == 1:
            dim = 0
        else:
            dim = 1
        prob = nnf.softmax(outputs, dim=dim)
        probs, classes = prob.topk(prob.shape[1])

        return probs.cpu().numpy(), classes.cpu().numpy()


class LIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_x, dataset_y):
        self.dataset_x, self.dataset_y = data_to_torch(dataset_x, dataset_y)

    def __len__(self):
        return self.dataset_x.shape[0]

    def __getitem__(self, idx):
        return self.dataset_x[idx - 1], self.dataset_y[idx - 1]
