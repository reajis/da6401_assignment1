from argparse import Namespace
import numpy as np
from .neural_layer import NeuralLayer


class NeuralNetwork:
    def __init__(
        self,
        input_size=784,
        hidden_sizes=None,
        num_layers=None,
        output_size=10,
        activation="relu",
        weight_init="random"
    ):
        # Allow config dict / Namespace
        if isinstance(input_size, Namespace):
            cfg = vars(input_size)
        elif isinstance(input_size, dict):
            cfg = input_size
        else:
            cfg = None

        if cfg is not None:
            input_size = cfg.get("input_size", cfg.get("input_dim", 784))
            hidden_sizes = cfg.get("hidden_sizes", cfg.get("hidden_size", hidden_sizes))
            num_layers = cfg.get("num_layers", cfg.get("nhl", num_layers))
            output_size = cfg.get("output_size", cfg.get("output_dim", 10))
            activation = cfg.get("activation", activation)
            weight_init = cfg.get("weight_init", weight_init)

        # normalize hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = []
        elif isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif isinstance(hidden_sizes, str):
            hidden_sizes = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
        else:
            hidden_sizes = list(hidden_sizes)

        if num_layers is None:
            num_layers = len(hidden_sizes)

        if num_layers == 0:
            hidden_sizes = []
        elif len(hidden_sizes) == 1 and num_layers > 1:
            hidden_sizes = hidden_sizes * num_layers
        elif len(hidden_sizes) != num_layers:
            raise ValueError("num_layers must match length of hidden_sizes")

        self.input_size = int(input_size)
        self.hidden_sizes = [int(h) for h in hidden_sizes]
        self.num_layers = int(num_layers)
        self.output_size = int(output_size)
        self.activation = activation
        self.weight_init = weight_init

        dims = [self.input_size] + self.hidden_sizes + [self.output_size]

        self.layers = []
        for i in range(len(dims) - 1):
            act = self.activation if i < len(dims) - 2 else "softmax"
            self.layers.append(
                NeuralLayer(
                    input_size=dims[i],
                    output_size=dims[i + 1],
                    activation=act,
                    weight_init=self.weight_init
                )
            )