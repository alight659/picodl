import numpy as np


class Tensor:

    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def _wrap(self, other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def _accumulate(self, grad):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        while grad.ndim > self.data.ndim:
            grad = grad.sum(axis=0)
        for i, dim in enumerate(self.data.shape):
            if dim == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        self.grad += grad

    def __add__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                other._accumulate(out.grad)
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self._accumulate(-out.grad)
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __mul__(self, other):
        other = self._wrap(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * other.data)
            if other.requires_grad:
                other._accumulate(out.grad * self.data)
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        out = Tensor(self.data ** power, self.requires_grad, (self,), f"**{power}")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * power * (self.data ** (power - 1)))
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = self._wrap(other)
        return self * (other ** -1)

    def matmul(self, other):
        other = self._wrap(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad @ other.data.swapaxes(-1, -2))
            if other.requires_grad:
                other._accumulate(self.data.swapaxes(-1, -2) @ out.grad)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def sum(self, axis=None):
        out = Tensor(self.data.sum(axis=axis), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    grad = np.expand_dims(grad, axis)
                self._accumulate(np.ones_like(self.data) * grad)
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (1 - t ** 2))
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / self.data)
        out._backward = _backward
        return out

    def clip(self, min_val, max_val):
        out = Tensor(np.clip(self.data, min_val, max_val), self.requires_grad, (self,), "clip")

        def _backward():
            if self.requires_grad:
                mask = (self.data >= min_val) & (self.data <= max_val)
                self._accumulate(out.grad * mask)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad * (self.data > 0))
        out._backward = _backward
        return out

    def mean(self, axis=None):
        out = Tensor(self.data.mean(axis=axis), self.requires_grad, (self,), "mean")
        n = self.data.size if axis is None else self.data.shape[axis]

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    grad = np.expand_dims(grad, axis)
                self._accumulate(np.ones_like(self.data) * grad / n)
        out._backward = _backward
        return out

    def sqrt(self):
        s = np.sqrt(self.data)
        out = Tensor(s, self.requires_grad, (self,), "sqrt")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad / (2 * s))
        out._backward = _backward
        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], self.requires_grad, (self,), "getitem")

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, out.grad)
                self._accumulate(grad)
        out._backward = _backward
        return out

    def conv2d(self, kernel, stride=1, padding=0, pad_mode="constant"):
        other = self._wrap(kernel)
        x = self.data
        w = other.data
        N, C, H, W = x.shape
        out_c, _, kh, kw = w.shape

        np_mode = {"constant": "constant", "reflect": "reflect", "replicate": "edge", "circular": "wrap"}[pad_mode]
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode=np_mode)

        Hp, Wp = x.shape[2], x.shape[3]
        out_h = (Hp - kh) // stride + 1
        out_w = (Wp - kw) // stride + 1

        # im2col
        cols = np.zeros((N, C, kh, kw, out_h, out_w))
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = x[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride]
        cols_reshaped = cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        w_reshaped = w.reshape(out_c, -1)
        out_data = cols_reshaped @ w_reshaped.T
        out_data = out_data.reshape(N, out_h, out_w, out_c).transpose(0, 3, 1, 2)

        out = Tensor(out_data, self.requires_grad or other.requires_grad, (self, other), "conv2d")

        def _backward():
            grad_out = out.grad.transpose(0, 2, 3, 1).reshape(N * out_h * out_w, out_c)
            if other.requires_grad:
                dw = grad_out.T @ cols_reshaped
                other._accumulate(dw.reshape(w.shape))
            if self.requires_grad:
                dcols = grad_out @ w_reshaped
                dcols = dcols.reshape(N, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
                dx = np.zeros_like(x)
                for i in range(kh):
                    for j in range(kw):
                        dx[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += dcols[:, :, i, j, :, :]
                if padding > 0:
                    dx = dx[:, :, padding:-padding, padding:-padding]
                self._accumulate(dx)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad.reshape(orig_shape))
        out._backward = _backward
        return out

    def view(self, *shape):
        return self.reshape(*shape)

    def flatten(self, start_dim=0):
        shape = self.data.shape
        new_shape = shape[:start_dim] + (-1,)
        return self.reshape(new_shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        if not axes:
            axes = tuple(reversed(range(self.data.ndim)))
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                inv = np.argsort(axes)
                self._accumulate(out.grad.transpose(inv))
        out._backward = _backward
        return out

    def permute(self, *axes):
        return self.transpose(*axes)

    def softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        s = exp / exp.sum(axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,), "softmax")

        def _backward():
            if self.requires_grad:
                grad_out = out.grad
                dot = np.sum(grad_out * s, axis=axis, keepdims=True)
                self._accumulate(s * (grad_out - dot))
        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        result = shifted - logsumexp
        out = Tensor(result, self.requires_grad, (self,), "log_softmax")

        def _backward():
            if self.requires_grad:
                s = np.exp(result)  # softmax probs
                grad_out = out.grad
                sum_grad = grad_out.sum(axis=axis, keepdims=True)
                self._accumulate(grad_out - s * sum_grad)
        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        c = np.sqrt(2 / np.pi)
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        result = 0.5 * x * (1 + t)
        out = Tensor(result, self.requires_grad, (self,), "gelu")

        def _backward():
            if self.requires_grad:
                dtanh = 1 - t ** 2
                dinner = c * (1 + 3 * 0.044715 * x ** 2)
                dresult = 0.5 * (1 + t) + 0.5 * x * dtanh * dinner
                self._accumulate(out.grad * dresult)
        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=True)
        mask = (self.data == out_data).astype(np.float64)
        # break ties: only count first occurrence
        count = mask.sum(axis=axis, keepdims=True)
        mask = mask / count
        if not keepdims:
            out_data = out_data.squeeze(axis=axis) if axis is not None else out_data.reshape(())
        out = Tensor(out_data, self.requires_grad, (self,), "max")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                self._accumulate(mask * grad)
        out._backward = _backward
        return out

    def argmax(self, axis=None):
        return np.argmax(self.data, axis=axis)

    def multinomial(self, num_samples=1):
        probs = self.data
        if probs.ndim == 1:
            probs = probs[None, :]
        out = np.zeros((probs.shape[0], num_samples), dtype=np.int64)
        for i in range(probs.shape[0]):
            out[i] = np.random.choice(probs.shape[1], size=num_samples, p=probs[i])
        return out

    @staticmethod
    def cat(tensors, axis=0):
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        datas = [t.data for t in tensors]
        out_data = np.concatenate(datas, axis=axis)
        req = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, req, tuple(tensors), "cat")
        sizes = [d.shape[axis] for d in datas]

        def _backward():
            idx = 0
            for t, size in zip(tensors, sizes):
                if t.requires_grad:
                    sl = [slice(None)] * out.grad.ndim
                    sl[axis] = slice(idx, idx + size)
                    t._accumulate(out.grad[tuple(sl)])
                idx += size
        out._backward = _backward
        return out

    @staticmethod
    def stack(tensors, axis=0):
        tensors = [t if isinstance(t, Tensor) else Tensor(t) for t in tensors]
        out_data = np.stack([t.data for t in tensors], axis=axis)
        req = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, req, tuple(tensors), "stack")

        def _backward():
            for i, t in enumerate(tensors):
                if t.requires_grad:
                    sl = [slice(None)] * out.grad.ndim
                    sl[axis] = i
                    t._accumulate(out.grad[tuple(sl)])
        out._backward = _backward
        return out

    def split(self, size_or_sections, axis=0):
        dim = self.data.shape[axis]
        if isinstance(size_or_sections, int):
            starts = list(range(0, dim, size_or_sections))
            sizes = [min(size_or_sections, dim - s) for s in starts]
        else:
            sizes = list(size_or_sections)
            starts = np.cumsum([0] + sizes[:-1]).tolist()
        outs = []
        for start, size in zip(starts, sizes):
            sl = [slice(None)] * self.data.ndim
            sl[axis] = slice(start, start + size)
            piece = Tensor(self.data[tuple(sl)], self.requires_grad, (self,), "split")

            def make_backward(sl_local, piece_local):
                def _backward():
                    if self.requires_grad:
                        grad = np.zeros_like(self.data)
                        grad[tuple(sl_local)] = piece_local.grad
                        self._accumulate(grad)
                return _backward
            piece._backward = make_backward(sl, piece)
            outs.append(piece)
        return outs

    def topk(self, k, axis=-1):
        idx = np.argsort(-self.data, axis=axis)
        idx = np.take(idx, range(k), axis=axis)
        values_data = np.take_along_axis(self.data, idx, axis=axis)
        out = Tensor(values_data, self.requires_grad, (self,), "topk")

        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                np.put_along_axis(grad, idx, out.grad, axis=axis)
                self._accumulate(grad)
        out._backward = _backward
        return out, idx

    def scatter(self, axis, index, src):
        src_t = src if isinstance(src, Tensor) else Tensor(src)
        out_data = self.data.copy()
        np.put_along_axis(out_data, index, src_t.data, axis=axis)
        out = Tensor(out_data, self.requires_grad or src_t.requires_grad, (self, src_t), "scatter")

        def _backward():
            if src_t.requires_grad:
                grad_src = np.take_along_axis(out.grad, index, axis=axis)
                src_t._accumulate(grad_src)
            if self.requires_grad:
                grad_self = out.grad.copy()
                np.put_along_axis(grad_self, index, 0, axis=axis)
                self._accumulate(grad_self)
        out._backward = _backward
        return out

    def index_add(self, axis, index, src):
        src_t = src if isinstance(src, Tensor) else Tensor(src)
        out_data = self.data.copy()
        np.add.at(out_data, (tuple(slice(None) for _ in range(axis)) + (index,)), src_t.data)
        out = Tensor(out_data, self.requires_grad or src_t.requires_grad, (self, src_t), "index_add")

        def _backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if src_t.requires_grad:
                sl = (tuple(slice(None) for _ in range(axis)) + (index,))
                src_t._accumulate(out.grad[sl])
        out._backward = _backward
        return out

    def maxpool2d(self, kernel_size, stride=None):
        stride = stride or kernel_size
        x = self.data
        N, C, H, W = x.shape
        kh = kw = kernel_size
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

        cols = np.zeros((N, C, kh, kw, out_h, out_w))
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = x[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride]
        cols_flat = cols.reshape(N, C, kh * kw, out_h, out_w)
        out_data = cols_flat.max(axis=2)
        argmax = cols_flat.argmax(axis=2)  # (N,C,out_h,out_w)

        out = Tensor(out_data, self.requires_grad, (self,), "maxpool2d")

        def _backward():
            if self.requires_grad:
                dx = np.zeros_like(x)
                grad = out.grad
                for i in range(kh):
                    for j in range(kw):
                        mask = (argmax == (i * kw + j)).astype(np.float64)
                        dx[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += mask * grad
                self._accumulate(dx)
        out._backward = _backward
        return out

    def avgpool2d(self, kernel_size, stride=None):
        stride = stride or kernel_size
        x = self.data
        N, C, H, W = x.shape
        kh = kw = kernel_size
        out_h = (H - kh) // stride + 1
        out_w = (W - kw) // stride + 1

        cols = np.zeros((N, C, kh, kw, out_h, out_w))
        for i in range(kh):
            for j in range(kw):
                cols[:, :, i, j, :, :] = x[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride]
        out_data = cols.mean(axis=(2, 3))

        out = Tensor(out_data, self.requires_grad, (self,), "avgpool2d")

        def _backward():
            if self.requires_grad:
                dx = np.zeros_like(x)
                grad = out.grad / (kh * kw)
                for i in range(kh):
                    for j in range(kw):
                        dx[:, :, i:i + stride * out_h:stride, j:j + stride * out_w:stride] += grad
                self._accumulate(dx)
        out._backward = _backward
        return out

    def backward(self, grad=None):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        if grad is None:
            grad = np.ones_like(self.data)
        self.grad = grad

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"