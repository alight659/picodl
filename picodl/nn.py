from typing import Sequence, Iterator
import numpy as np
from picodl.tensor import Tensor
from picodl.layers import Layer


class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def params(self) -> Iterator[Tensor]:
        for layer in self.layers:
            for param in layer.params.values():
                yield param

    def zero_grad(self) -> None:
        for param in self.params():
            param.zero_grad()

    def save(self, path: str) -> None:
        flat = {}
        for i, layer in enumerate(self.layers):
            for name, param in layer.params.items():
                flat[f"{i}_{name}"] = param.data
        with open(path, "wb") as f:
            np.savez(f, **flat)

    def load(self, path: str) -> None:
        data = np.load(path)
        for i, layer in enumerate(self.layers):
            for name, param in layer.params.items():
                key = f"{i}_{name}"
                if key in data:
                    param.data = data[key]
                else:
                    raise KeyError(f"missing {key} in {path}, architecture mismatch")