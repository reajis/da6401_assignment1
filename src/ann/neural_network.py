from argparse import Namespace
from .neural_layer import NeuralLayer


class NeuralNetwork:
    def __init__(self,
                 input_size=784,
                 hidden_sizes=None,
                 num_layers=None,
                 output_size=10,
                 activation="relu",
                 weight_init="random"):

        # Handle case: NeuralNetwork(args) where args is argparse.Namespace
        if isinstance(input_size, Namespace):
            args = input_size

            input_size = getattr(args, "input_size", 784)
            hidden_sizes = getattr(args, "hidden_sizes", getattr(args, "hidden_size", None))
            num_layers = getattr(args, "num_layers", None)
            output_size = getattr(args, "output_size", 10)
            activation = getattr(args, "activation", "relu")
            weight_init = getattr(args, "weight_init", "random")

        self.layers = []

        # Normalize hidden_sizes
        if hidden_sizes is None:
            hidden_sizes = []
        elif isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif isinstance(hidden_sizes, str):
            # Supports formats like "128,64,32"
            hidden_sizes = [int(x.strip()) for x in hidden_sizes.split(",") if x.strip()]
        else:
            hidden_sizes = list(hidden_sizes)

        # Infer num_layers if not provided
        if num_layers is None:
            num_layers = len(hidden_sizes)

        # Ensure integer type
        num_layers = int(num_layers)
        input_size = int(input_size)
        output_size = int(output_size)

        # Make hidden_sizes compatible with num_layers
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

        # No hidden layers
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
            # First hidden layer
            self.layers.append(
                NeuralLayer(
                    input_size=input_size,
                    output_size=hidden_sizes[0],
                    activation=activation,
                    weight_init=weight_init
                )
            )

            # Remaining hidden layers
            for i in range(1, num_layers):
                self.layers.append(
                    NeuralLayer(
                        input_size=hidden_sizes[i - 1],
                        output_size=hidden_sizes[i],
                        activation=activation,
                        weight_init=weight_init
                    )
                )

            # Output layer
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