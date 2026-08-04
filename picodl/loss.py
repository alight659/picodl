import numpy as np
from picodl.tensor import Tensor


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        diff = predicted - actual
        return (diff * diff).sum()


class BinaryCrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> Tensor:
        eps = 1e-9
        p = predicted.clip(eps, 1 - eps)
        term1 = actual * p.log()
        term2 = (1 - actual) * (1 - p).log()
        return -(term1 + term2).sum()


class NLLLoss(Loss):
    # Expects predicted = log-probabilities (N, C), actual = int class indices (N,)
    def loss(self, predicted: Tensor, actual) -> Tensor:
        n = predicted.shape[0]
        picked = predicted[np.arange(n), actual]
        return -picked.mean()


class CrossEntropyLoss(Loss):
    # Expects predicted = raw logits (N, C), actual = int class indices (N,). 
    def loss(self, predicted: Tensor, actual) -> Tensor:
        log_probs = predicted.log_softmax(axis=-1)
        n = predicted.shape[0]
        picked = log_probs[np.arange(n), actual]
        return -picked.mean()