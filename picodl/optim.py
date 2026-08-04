import numpy as np
from picodl.nn import NeuralNet


class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param in net.params():
            if param.grad is not None:
                param.data -= self.lr * param.grad


class RMSprop(Optimizer):
    def __init__(self, lr: float = 0.001, decay: float = 0.9, eps: float = 1e-8) -> None:
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}

    def step(self, net: NeuralNet) -> None:
        for param in net.params():
            if param.grad is None:
                continue
            key = id(param)
            if key not in self.cache:
                self.cache[key] = np.zeros_like(param.data)
            self.cache[key] = self.decay * self.cache[key] + (1 - self.decay) * (param.grad ** 2)
            param.data -= self.lr * param.grad / (np.sqrt(self.cache[key]) + self.eps)


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, net: NeuralNet) -> None:
        self.t += 1
        for param in net.params():
            if param.grad is None:
                continue
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param.data)
                self.v[key] = np.zeros_like(param.data)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * param.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    # Adam with decoupled weight decay (weight decay applied directly to param.data, not folded into the gradient/momentum like plain Adam+L2)
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, net: NeuralNet) -> None:
        self.t += 1
        for param in net.params():
            if param.grad is None:
                continue
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param.data)
                self.v[key] = np.zeros_like(param.data)
            # decoupled weight decay: shrink weights directly, not via gradient
            param.data -= self.lr * self.weight_decay * param.data
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * param.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)