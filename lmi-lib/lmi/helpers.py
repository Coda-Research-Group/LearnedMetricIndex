import json
import torch.nn as nn


def extract_model_config(model: nn.Sequential):
    layers = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            layers.append(
                {
                    "type": "linear",
                    "fanin": layer.in_features,
                    "fanout": layer.out_features,
                }
            )
        elif isinstance(layer, nn.ReLU):
            layers.append({"type": "relu"})
        else:
            raise NotImplementedError(f"Layer type {type(layer)} not supported")

    return json.dumps(layers)
