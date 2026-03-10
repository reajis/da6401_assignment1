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

        cfg = None
        if isinstance(input_size, Namespace):
            cfg = vars(input_size)
        elif isinstance(input_size, dict):
            cfg = input_size

        if cfg is not None:
            input_size = cfg.get("input_size", cfg.get("input_dim", cfg.get("n_input", 784)))
            hidden_sizes = cfg.get(
                "hidden_sizes",
                cfg.get("hidden_size", cfg.get("sizes", hidden_sizes))
            )
            num_layers = cfg.get("num_layers", cfg.get("nhl", cfg.get("n_hidden_layers", num_layers)))
            output_size = cfg.get(
                "output_size",
                cfg.get("output_dim", cfg.get("num_classes", cfg.get("n_classes", 10)))
            )
            activation = cfg.get("activation", activation)
            weight_init = cfg.get("weight_init", weight_init)

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

        input_size = int(input_size)
        output_size = int(output_size)
        num_layers = int(num_layers)

        if num_layers == 0:
            hidden_sizes = []
        else:
            if len(hidden_sizes) == 0:
                raise ValueError("hidden_sizes must be provided when num_layers > 0")
            if len(hidden_sizes) == 1 and num_layers > 1:
                hidden_sizes = hidden_sizes * num_layers
            elif len(hidden_sizes) != num_layers:
                raise ValueError("num_layers must match length of hidden_sizes")

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.output_size = output_size
        self.activation = activation
        self.weight_init = weight_init
        self.layers = []

        self._build_layers_from_dims([self.input_size] + self.hidden_sizes + [self.output_size])

    def _build_layers_from_dims(self, dims):
        if len(dims) < 2:
            raise ValueError("dims must contain at least input and output dimensions")

        self.input_size = int(dims[0])
        self.output_size = int(dims[-1])
        self.hidden_sizes = [int(x) for x in dims[1:-1]]
        self.num_layers = len(self.hidden_sizes)

        self.layers = []
        for i in range(len(dims) - 1):
            act = self.activation if i < len(dims) - 2 else "softmax"
            self.layers.append(
                NeuralLayer(
                    input_size=int(dims[i]),
                    output_size=int(dims[i + 1]),
                    activation=act,
                    weight_init=self.weight_init
                )
            )

    def forward(self, X):
        activations = [X]
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            activations.append(out)
        return out, activations

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_layers(self):
        return self.layers

    def get_weights(self):
        weights = {}
        for i, layer in enumerate(self.layers, start=1):
            weights[f"W{i}"] = layer.W.copy()
            weights[f"b{i}"] = layer.b.copy()
        return weights

    def set_weights(self, weights):
        if isinstance(weights, np.ndarray) and weights.shape == ():
            weights = weights.item()

        if isinstance(weights, (list, tuple)):
            converted = {}
            for i, item in enumerate(weights, start=1):
                if not isinstance(item, dict) or "W" not in item or "b" not in item:
                    raise ValueError("List format must be [{'W':..., 'b':...}, ...]")
                converted[f"W{i}"] = item["W"]
                converted[f"b{i}"] = item["b"]
            weights = converted

        if not isinstance(weights, dict):
            raise ValueError(f"Expected weights to be a dict, got {type(weights).__name__}")

        layer_ids = sorted(
            int(k[1:]) for k in weights.keys()
            if k.startswith("W") and k[1:].isdigit()
        )

        if not layer_ids:
            raise ValueError("No weight keys like W1, W2, ... found in weights dict")

        dims = []
        processed = {}

        prev_out_dim = None
        for i in layer_ids:
            w_key = f"W{i}"
            b_key = f"b{i}"

            if w_key not in weights or b_key not in weights:
                raise ValueError(f"Missing {w_key} or {b_key} in weights dict")

            W = np.array(weights[w_key], dtype=float, copy=True)
            b = np.array(weights[b_key], dtype=float, copy=True)

            if W.ndim != 2:
                raise ValueError(f"{w_key} must be 2D, got shape {W.shape}")

            if b.ndim == 1:
                b = b.reshape(1, -1)
            elif b.ndim == 2 and b.shape[1] == 1:
                b = b.T

            if b.shape != (1, W.shape[1]):
                raise ValueError(
                    f"{b_key} shape {b.shape} incompatible with {w_key} shape {W.shape}"
                )

            if prev_out_dim is not None and W.shape[0] != prev_out_dim:
                raise ValueError(
                    f"Inconsistent layer shapes: previous output dim {prev_out_dim}, "
                    f"but {w_key} has input dim {W.shape[0]}"
                )

            if not dims:
                dims.append(W.shape[0])
            dims.append(W.shape[1])

            prev_out_dim = W.shape[1]
            processed[i] = (W, b)

        rebuild_needed = (
            len(self.layers) != len(layer_ids) or
            any(
                self.layers[idx - 1].W.shape != processed[idx][0].shape
                for idx in layer_ids
                if idx - 1 < len(self.layers)
            )
        )

        if rebuild_needed:
            self._build_layers_from_dims(dims)

        for i in layer_ids:
            self.layers[i - 1].W = processed[i][0]
            self.layers[i - 1].b = processed[i][1]