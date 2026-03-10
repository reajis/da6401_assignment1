import numpy as np
from ann.activations import softmax
from ann.neural_layer import DenseLayer
from ann.objective_functions import cross_entropy_loss, cross_entropy_grad
from ann.objective_functions import mse_loss, mse_grad
from ann.optimizers import Optimizer


class NeuralNetwork:
    def __init__(
        self,
        cli_args=None,
        input_dim=None,
        hidden_layers=None,
        output_dim=None,
        activation="relu",
        loss="cross_entropy",
        weight_init="xavier",
        learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=0.0,
    ):
        # Read settings from CLI args if provided
        if cli_args is not None:
            input_dim = 784
            output_dim = 10

            num_layers = getattr(cli_args, "num_layers", getattr(cli_args, "hidden_layers", 1))
            hidden_size = getattr(cli_args, "hidden_size", getattr(cli_args, "num_neurons", [64]))

            if isinstance(hidden_size, int):
                hidden_layers = [hidden_size] * num_layers
            else:
                hidden_layers = list(hidden_size)
                if len(hidden_layers) == 1:
                    hidden_layers = hidden_layers * num_layers

            activation = getattr(cli_args, "activation", activation)
            loss = getattr(cli_args, "loss", loss)
            weight_init = getattr(cli_args, "weight_init", weight_init)
            learning_rate = getattr(cli_args, "learning_rate", learning_rate)
            optimizer_name = getattr(cli_args, "optimizer", optimizer_name)
            weight_decay = getattr(cli_args, "weight_decay", weight_decay)

        if input_dim is None:
            input_dim = 784
        if output_dim is None:
            output_dim = 10
        if hidden_layers is None:
            hidden_layers = [64]

        self.layers = []
        layer_dims = [input_dim] + hidden_layers + [output_dim]

        for idx in range(len(layer_dims) - 1):
            layer_activation = activation if idx < len(layer_dims) - 2 else None
            layer = DenseLayer(
                layer_dims[idx],
                layer_dims[idx + 1],
                activation=layer_activation,
                weight_init=weight_init,
            )
            self.layers.append(layer)

        self.loss = loss
        self.weight_decay = weight_decay
        self.optimizer = Optimizer(name=optimizer_name, learning_rate=learning_rate)
        self.optimizer.setup(self.layers)
        self.grad_W = None
        self.grad_b = None
        self.last_hidden_output = None

    def forward(self, X):
        layer_output = X
        for idx, layer in enumerate(self.layers):
            layer_output = layer.forward(layer_output)
            if idx == 0:
                self.last_hidden_output = layer_output.copy()
        return layer_output

    def compute_loss(self, logits, y_true):
        if self.loss == "cross_entropy":
            return cross_entropy_loss(logits, y_true)
        return mse_loss(logits, y_true)

    def backward(self, y_true, y_pred):
        if self.loss == "cross_entropy":
            grad = cross_entropy_grad(y_pred, y_true)
        else:
            grad = mse_grad(y_pred, y_true)

        grad_W_list = []
        grad_b_list = []

        grad = self.layers[-1].backward_linear(grad, self.weight_decay)
        grad_W_list.append(self.layers[-1].grad_W.copy())
        grad_b_list.append(self.layers[-1].grad_b.copy())

        for layer in reversed(self.layers[:-1]):
            grad = layer.backward(grad, self.weight_decay)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        self.grad_W = grad_W_list
        self.grad_b = grad_b_list
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step(self.layers)

    def train_batch(self, X, y):
        logits = self.forward(X)
        loss_value = self.compute_loss(logits, y)
        self.backward(y, logits)
        self.update_weights()
        return loss_value

    def predict_proba(self, X):
        logits = self.forward(X)
        return softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        logits = self.forward(X)
        loss_value = self.compute_loss(logits, y)
        preds = np.argmax(softmax(logits), axis=1)
        accuracy = np.mean(preds == y)
        return loss_value, accuracy, preds

    def get_weights(self):
        weight_dict = {}
        for idx, layer in enumerate(self.layers):
            weight_dict[f"W{idx}"] = layer.W.copy()
            weight_dict[f"b{idx}"] = layer.b.copy()
        return weight_dict

    def set_weights(self, weight_dict):
        for idx, layer in enumerate(self.layers):
            w_key = f"W{idx}"
            b_key = f"b{idx}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()