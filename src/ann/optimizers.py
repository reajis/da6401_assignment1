import numpy as np


class Optimizer:
    def __init__(
        self,
        name="sgd",
        learning_rate=0.001,
        momentum=0.9,
        beta=0.9,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ):
        # Optimizer settings
        self.name = name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.v_W = []
        self.v_b = []
        self.m_W = []
        self.m_b = []

    def setup(self, layers):
        # State per layer
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.m_W = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]

    def _check(self, layers):
        # Rebuild state if needed
        if len(self.v_W) != len(layers):
            self.setup(layers)

    def step(self, layers):
        # Single update step
        self._check(layers)
        self.t += 1

        if self.name == "sgd":
            for layer in layers:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b

        elif self.name == "momentum":
            for idx, layer in enumerate(layers):
                self.v_W[idx] = self.momentum * self.v_W[idx] - self.learning_rate * layer.grad_W
                self.v_b[idx] = self.momentum * self.v_b[idx] - self.learning_rate * layer.grad_b
                layer.W += self.v_W[idx]
                layer.b += self.v_b[idx]

        elif self.name == "nag":
            for idx, layer in enumerate(layers):
                prev_v_W = self.v_W[idx].copy()
                prev_v_b = self.v_b[idx].copy()

                self.v_W[idx] = self.momentum * self.v_W[idx] - self.learning_rate * layer.grad_W
                self.v_b[idx] = self.momentum * self.v_b[idx] - self.learning_rate * layer.grad_b

                layer.W += -self.momentum * prev_v_W + (1 + self.momentum) * self.v_W[idx]
                layer.b += -self.momentum * prev_v_b + (1 + self.momentum) * self.v_b[idx]

        elif self.name == "rmsprop":
            for idx, layer in enumerate(layers):
                self.v_W[idx] = self.beta * self.v_W[idx] + (1.0 - self.beta) * (layer.grad_W ** 2)
                self.v_b[idx] = self.beta * self.v_b[idx] + (1.0 - self.beta) * (layer.grad_b ** 2)

                layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.v_W[idx]) + self.epsilon)
                layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.v_b[idx]) + self.epsilon)

        elif self.name == "adam":
            for idx, layer in enumerate(layers):
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1.0 - self.beta1) * layer.grad_W
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * layer.grad_b
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (layer.grad_W ** 2)
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (layer.grad_b ** 2)

                m_W_hat = self.m_W[idx] / (1.0 - self.beta1 ** self.t)
                m_b_hat = self.m_b[idx] / (1.0 - self.beta1 ** self.t)
                v_W_hat = self.v_W[idx] / (1.0 - self.beta2 ** self.t)
                v_b_hat = self.v_b[idx] / (1.0 - self.beta2 ** self.t)

                layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        elif self.name == "nadam":
            for idx, layer in enumerate(layers):
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1.0 - self.beta1) * layer.grad_W
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * layer.grad_b
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (layer.grad_W ** 2)
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (layer.grad_b ** 2)

                m_W_hat = self.m_W[idx] / (1.0 - self.beta1 ** self.t)
                m_b_hat = self.m_b[idx] / (1.0 - self.beta1 ** self.t)
                v_W_hat = self.v_W[idx] / (1.0 - self.beta2 ** self.t)
                v_b_hat = self.v_b[idx] / (1.0 - self.beta2 ** self.t)

                nesterov_W = self.beta1 * m_W_hat + ((1.0 - self.beta1) * layer.grad_W) / (1.0 - self.beta1 ** self.t)
                nesterov_b = self.beta1 * m_b_hat + ((1.0 - self.beta1) * layer.grad_b) / (1.0 - self.beta1 ** self.t)

                layer.W -= self.learning_rate * nesterov_W / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * nesterov_b / (np.sqrt(v_b_hat) + self.epsilon)

        else:
            raise ValueError("Unknown optimizer")