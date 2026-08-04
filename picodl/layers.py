import numpy as np
from typing import Dict
from picodl.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = Tensor(np.random.randn(input_size, output_size) * 0.1, requires_grad=True)
        self.params["b"] = Tensor(np.random.randn(output_size) * 0.1, requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.matmul(self.params["w"]) + self.params["b"]


class Tanh(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()


class ReLU(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.relu()


class GELU(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.gelu()


class Conv2D(Layer):
    # inputs: (N, C, H, W). kernel: (out_c, C, kh, kw)
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, pad_mode: str = "constant") -> None:
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.pad_mode = pad_mode
        scale = 0.1
        self.params["w"] = Tensor(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale, requires_grad=True)
        self.params["b"] = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs.conv2d(self.params["w"], stride=self.stride, padding=self.padding, pad_mode=self.pad_mode)
        return self._add_bias(out, self.params["b"])

    def _add_bias(self, out: Tensor, b: Tensor) -> Tensor:
        bias = Tensor(b.data.reshape(1, -1, 1, 1), b.requires_grad, (b,), "reshape_bias")

        def _backward():
            if b.requires_grad:
                b._accumulate(bias.grad.sum(axis=(0, 2, 3)))
        bias._backward = _backward
        return out + bias


class Embedding(Layer):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.params["w"] = Tensor(np.random.randn(vocab_size, embed_dim) * 0.1, requires_grad=True)

    def forward(self, idx) -> Tensor:
        return self.params["w"][idx]


class TiedLinear(Layer):
    def __init__(self, tied_layer: Layer, weight_key: str = "w", bias: bool = False) -> None:
        super().__init__()
        self.tied_weight = tied_layer.params[weight_key]  # shape (vocab, dim), shared object
        if bias:
            out_features = self.tied_weight.data.shape[0]
            self.params["b"] = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        out = inputs.matmul(self.tied_weight.transpose(1, 0))
        if "b" in self.params:
            out = out + self.params["b"]
        return out


class MaxPool2D(Layer):
    def __init__(self, kernel_size: int, stride: int = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.maxpool2d(self.kernel_size, self.stride)


class AvgPool2D(Layer):
    def __init__(self, kernel_size: int, stride: int = None) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.avgpool2d(self.kernel_size, self.stride)


class Flatten(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.flatten(start_dim=1)


class GlobalAvgPool(Layer):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs.mean(axis=3).mean(axis=2)             # (N,C,H,W) -> (N,C)


class BatchNorm2D(Layer):
    # Normalize over N,H,W per channel. inputs: (N,C,H,W)
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.params["gamma"] = Tensor(np.ones(num_features), requires_grad=True)
        self.params["beta"] = Tensor(np.zeros(num_features), requires_grad=True)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True

    def forward(self, inputs: Tensor) -> Tensor:
        if self.training:
            mean = inputs.data.mean(axis=(0, 2, 3))
            var = inputs.data.var(axis=(0, 2, 3))
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        mean_t = Tensor(mean.reshape(1, -1, 1, 1))
        var_t = Tensor(var.reshape(1, -1, 1, 1))
        gamma = self.params["gamma"]
        beta = self.params["beta"]
        gamma_b = Tensor(gamma.data.reshape(1, -1, 1, 1), gamma.requires_grad, (gamma,), "reshape_g")

        def _bg():
            if gamma.requires_grad:
                gamma._accumulate(gamma_b.grad.sum(axis=(0, 2, 3)))
        gamma_b._backward = _bg

        beta_b = Tensor(beta.data.reshape(1, -1, 1, 1), beta.requires_grad, (beta,), "reshape_bt")

        def _bb():
            if beta.requires_grad:
                beta._accumulate(beta_b.grad.sum(axis=(0, 2, 3)))
        beta_b._backward = _bb

        normed = (inputs - mean_t) / (var_t + self.eps).sqrt()
        return normed * gamma_b + beta_b

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class Dropout(Layer):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, inputs: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return inputs
        mask = (np.random.rand(*inputs.data.shape) > self.p).astype(np.float64) / (1 - self.p)
        mask_t = Tensor(mask)
        return inputs * mask_t

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class LayerNorm(Layer):
    # y = (x - mean) / sqrt(var + eps) * gamma + beta
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.params["gamma"] = Tensor(np.ones(dim), requires_grad=True)
        self.params["beta"] = Tensor(np.zeros(dim), requires_grad=True)

    def forward(self, inputs: Tensor) -> Tensor:
        mu = inputs.mean(axis=-1)
        mu_b = Tensor(np.expand_dims(mu.data, -1), mu.requires_grad, (mu,), "expand")

        def _backward_mu():
            if mu.requires_grad:
                mu._accumulate(mu_b.grad.sum(axis=-1))
        mu_b._backward = _backward_mu

        diff = inputs - mu_b
        var = (diff * diff).mean(axis=-1)
        var_b = Tensor(np.expand_dims(var.data, -1), var.requires_grad, (var,), "expand")

        def _backward_var():
            if var.requires_grad:
                var._accumulate(var_b.grad.sum(axis=-1))
        var_b._backward = _backward_var

        std = (var_b + self.eps).sqrt()
        normed = diff / std
        return normed * self.params["gamma"] + self.params["beta"]