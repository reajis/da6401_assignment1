
import numpy as np


class Optimizer:
    def __init__(self, name="sgd", learning_rate=0.001, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
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
        self.v_W = [np.zeros_like(layer.W) for layer in layers]
        self.v_b = [np.zeros_like(layer.b) for layer in layers]
        self.m_W = [np.zeros_like(layer.W) for layer in layers]
        self.m_b = [np.zeros_like(layer.b) for layer in layers]

    def _check(self, layers):
        if len(self.v_W) != len(layers):
            self.setup(layers)

    def step(self, layers):
        self._check(layers)
        self.t += 1

        if self.name == "sgd":
            for layer in layers:
                layer.W -= self.learning_rate * layer.grad_W
                layer.b -= self.learning_rate * layer.grad_b

        elif self.name == "momentum":
            for i, layer in enumerate(layers):
                self.v_W[i] = self.momentum * self.v_W[i] - self.learning_rate * layer.grad_W
                self.v_b[i] = self.momentum * self.v_b[i] - self.learning_rate * layer.grad_b
                layer.W += self.v_W[i]
                layer.b += self.v_b[i]

        elif self.name == "nag":
            for i, layer in enumerate(layers):
                prev_v_W = self.v_W[i].copy()
                prev_v_b = self.v_b[i].copy()
                self.v_W[i] = self.momentum * self.v_W[i] - self.learning_rate * layer.grad_W
                self.v_b[i] = self.momentum * self.v_b[i] - self.learning_rate * layer.grad_b
                layer.W += -self.momentum * prev_v_W + (1 + self.momentum) * self.v_W[i]
                layer.b += -self.momentum * prev_v_b + (1 + self.momentum) * self.v_b[i]

        elif self.name == "rmsprop":
            for i, layer in enumerate(layers):
                self.v_W[i] = self.beta * self.v_W[i] + (1.0 - self.beta) * (layer.grad_W ** 2)
                self.v_b[i] = self.beta * self.v_b[i] + (1.0 - self.beta) * (layer.grad_b ** 2)
                layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.v_W[i]) + self.epsilon)
                layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.v_b[i]) + self.epsilon)

        elif self.name == "adam":
            for i, layer in enumerate(layers):
                self.m_W[i] = self.beta1 * self.m_W[i] + (1.0 - self.beta1) * layer.grad_W
                self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * layer.grad_b
                self.v_W[i] = self.beta2 * self.v_W[i] + (1.0 - self.beta2) * (layer.grad_W ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (layer.grad_b ** 2)

                m_W_hat = self.m_W[i] / (1.0 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1.0 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)

                layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        elif self.name == "nadam":
            for i, layer in enumerate(layers):
                self.m_W[i] = self.beta1 * self.m_W[i] + (1.0 - self.beta1) * layer.grad_W
                self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * layer.grad_b
                self.v_W[i] = self.beta2 * self.v_W[i] + (1.0 - self.beta2) * (layer.grad_W ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1.0 - self.beta2) * (layer.grad_b ** 2)

                m_W_hat = self.m_W[i] / (1.0 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1.0 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1.0 - self.beta2 ** self.t)

                nesterov_W = self.beta1 * m_W_hat + ((1.0 - self.beta1) * layer.grad_W) / (1.0 - self.beta1 ** self.t)
                nesterov_b = self.beta1 * m_b_hat + ((1.0 - self.beta1) * layer.grad_b) / (1.0 - self.beta1 ** self.t)

                layer.W -= self.learning_rate * nesterov_W / (np.sqrt(v_W_hat) + self.epsilon)
                layer.b -= self.learning_rate * nesterov_b / (np.sqrt(v_b_hat) + self.epsilon)

        else:
            raise ValueError("Unknown optimizer")
