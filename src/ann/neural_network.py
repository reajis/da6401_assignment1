from argparse import Namespace
from .neural_layer import NeuralLayer
import numpy as np



class NeuralNetwork:
    def __init__(self,
                 input_size=784,
                 hidden_sizes=None,
                 num_layers=None,
                 output_size=10,
                 activation="relu",
                 weight_init="random"):

        if isinstance(input_size, Namespace):
            args = input_size
            input_size = getattr(args, "input_size", 784)
            hidden_sizes = getattr(args, "hidden_sizes", getattr(args, "hidden_size", None))
            num_layers = getattr(args, "num_layers", None)
            output_size = getattr(args, "output_size", 10)
            activation = getattr(args, "activation", "relu")
            weight_init = getattr(args, "weight_init", "random")

        self.layers = []

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

        num_layers = int(num_layers)
        input_size = int(input_size)
        output_size = int(output_size)

        if num_layers == 0:
            hidden_sizes = []
        else:
            if len(hidden_sizes) == 0:
                raise ValueError("hidden_sizes must be provided when num_layers > 0")
            elif len(hidden_sizes) == 1 and num_layers > 1:
                hidden_sizes = hidden_sizes * num_layers
            elif len(hidden_sizes) != num_layers:
                raise ValueError("num_layers must match length of hidden_sizes")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init

        if num_layers == 0:
            self.layers.append(
                NeuralLayer(
                    input_size=input_size,
                    output_size=output_size,
                    activation="softmax",
                    weight_init=weight_init
                )
            )
        else:
            self.layers.append(
                NeuralLayer(
                    input_size=input_size,
                    output_size=hidden_sizes[0],
                    activation=activation,
                    weight_init=weight_init
                )
            )

            for i in range(1, num_layers):
                self.layers.append(
                    NeuralLayer(
                        input_size=hidden_sizes[i - 1],
                        output_size=hidden_sizes[i],
                        activation=activation,
                        weight_init=weight_init
                    )
                )

            self.layers.append(
                NeuralLayer(
                    input_size=hidden_sizes[-1],
                    output_size=output_size,
                    activation="softmax",
                    weight_init=weight_init
                )
            )

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dA):
        grad = dA
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_layers(self):
        return self.layers

    def set_weights(self, weights):
        """
        Supports:
        1. {"W1": ..., "b1": ..., "W2": ..., "b2": ..., ...}
        2. {"W": [W1, W2, ...], "b": [b1, b2, ...]}
        3. [(W1, b1), (W2, b2), ...]
        4. [W1, b1, W2, b2, ...]
        """

        # If loaded via np.load(..., allow_pickle=True), unwrap scalar object array
        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        normalized = []

        if isinstance(weights, dict):
            # Format: {"W1":..., "b1":..., "W2":..., "b2":...}
            if all(f"W{i}" in weights and f"b{i}" in weights for i in range(1, len(self.layers) + 1)):
                for i in range(1, len(self.layers) + 1):
                    normalized.append((weights[f"W{i}"], weights[f"b{i}"]))

            # Format: {"W":[...], "b":[...]}
            elif "W" in weights and "b" in weights:
                if len(weights["W"]) != len(self.layers) or len(weights["b"]) != len(self.layers):
                    raise ValueError("Weight dict lists must match number of layers")
                normalized = list(zip(weights["W"], weights["b"]))

            # Format: {"weights":[...], "biases":[...]}
            elif "weights" in weights and "biases" in weights:
                if len(weights["weights"]) != len(self.layers) or len(weights["biases"]) != len(self.layers):
                    raise ValueError("Weight dict lists must match number of layers")
                normalized = list(zip(weights["weights"], weights["biases"]))

            else:
                raise ValueError("Unsupported dictionary format for weights")

        else:
            weights = list(weights)

            # Format: [W1, b1, W2, b2, ...]
            if len(weights) == 2 * len(self.layers):
                normalized = [
                    (weights[i], weights[i + 1])
                    for i in range(0, len(weights), 2)
                ]

            # Format: [(W1,b1), (W2,b2), ...]
            elif len(weights) == len(self.layers):
                for item in weights:
                    if isinstance(item, dict):
                        if "W" in item and "b" in item:
                            normalized.append((item["W"], item["b"]))
                        else:
                            raise ValueError("Each layer dict must contain 'W' and 'b'")
                    else:
                        W, b = item
                        normalized.append((W, b))

            else:
                raise ValueError("Number of weight sets must match number of layers")

        for layer, (W, b) in zip(self.layers, normalized):
            layer.W = np.array(W, dtype=float, copy=True)
            layer.b = np.array(b, dtype=float, copy=True)

    def get_weights(self):
        """
        Return weights and biases of all layers in the format:
        [
            (W1, b1),
            (W2, b2),
            ...
        ]
        """
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]