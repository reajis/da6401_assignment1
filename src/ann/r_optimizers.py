"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np

class Optimizer:
    def __init__(
        self,
        layers,
        optimizer_type="sgd",
        lr=0.01,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.layers = layers
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr
        self.weight_decay = weight_decay

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.t = 0

        # first moment / velocity terms
        self.v_W = {id(layer): np.zeros_like(layer.W) for layer in layers}
        self.v_b = {id(layer): np.zeros_like(layer.b) for layer in layers}

        # second moment terms
        self.m_W = {id(layer): np.zeros_like(layer.W) for layer in layers}
        self.m_b = {id(layer): np.zeros_like(layer.b) for layer in layers}

    def step(self):
        self.t += 1

        for layer in self.layers:
            if getattr(layer, "W", None) is None:
                continue
            if layer.grad_W is None or layer.grad_b is None:
                continue

            idx = id(layer)

            grad_W = layer.grad_W.copy()
            grad_b = layer.grad_b.copy()

            # L2 weight decay on weights only
            if self.weight_decay > 0.0:
                grad_W += self.weight_decay * layer.W

            # SGD
            if self.optimizer_type == "sgd":
                layer.W -= self.lr * grad_W
                layer.b -= self.lr * grad_b

            # Momentum
            elif self.optimizer_type == "momentum":
                self.v_W[idx] = self.beta1 * self.v_W[idx] + grad_W
                self.v_b[idx] = self.beta1 * self.v_b[idx] + grad_b

                layer.W -= self.lr * self.v_W[idx]
                layer.b -= self.lr * self.v_b[idx]

            # NAG
            elif self.optimizer_type == "nag":
                vW_prev = self.v_W[idx].copy()
                vB_prev = self.v_b[idx].copy()

                self.v_W[idx] = self.beta1 * self.v_W[idx] + grad_W
                self.v_b[idx] = self.beta1 * self.v_b[idx] + grad_b

                layer.W -= self.lr * (grad_W + self.beta1 * vW_prev)
                layer.b -= self.lr * (grad_b + self.beta1 * vB_prev)

            # RMSProp
            elif self.optimizer_type == "rmsprop":
                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (grad_W ** 2)
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (grad_b ** 2)

                layer.W -= (self.lr * grad_W) / (np.sqrt(self.v_W[idx]) + self.epsilon)
                layer.b -= (self.lr * grad_b) / (np.sqrt(self.v_b[idx]) + self.epsilon)

            # Adam
            elif self.optimizer_type == "adam":
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1.0 - self.beta1) * grad_W
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * grad_b

                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (grad_W ** 2)
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (grad_b ** 2)

                mW_hat = self.m_W[idx] / (1.0 - self.beta1 ** self.t)
                mB_hat = self.m_b[idx] / (1.0 - self.beta1 ** self.t)

                vW_hat = self.v_W[idx] / (1.0 - self.beta2 ** self.t)
                vB_hat = self.v_b[idx] / (1.0 - self.beta2 ** self.t)

                layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.epsilon)
                layer.b -= self.lr * mB_hat / (np.sqrt(vB_hat) + self.epsilon)

            # Nadam
            elif self.optimizer_type == "nadam":
                self.m_W[idx] = self.beta1 * self.m_W[idx] + (1.0 - self.beta1) * grad_W
                self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * grad_b

                self.v_W[idx] = self.beta2 * self.v_W[idx] + (1.0 - self.beta2) * (grad_W ** 2)
                self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * (grad_b ** 2)

                mW_hat = self.m_W[idx] / (1.0 - self.beta1 ** self.t)
                mB_hat = self.m_b[idx] / (1.0 - self.beta1 ** self.t)

                vW_hat = self.v_W[idx] / (1.0 - self.beta2 ** self.t)
                vB_hat = self.v_b[idx] / (1.0 - self.beta2 ** self.t)

                mW_nesterov = self.beta1 * mW_hat + ((1.0 - self.beta1) * grad_W) / (1.0 - self.beta1 ** self.t)
                mB_nesterov = self.beta1 * mB_hat + ((1.0 - self.beta1) * grad_b) / (1.0 - self.beta1 ** self.t)

                layer.W -= self.lr * mW_nesterov / (np.sqrt(vW_hat) + self.epsilon)
                layer.b -= self.lr * mB_nesterov / (np.sqrt(vB_hat) + self.epsilon)

            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")