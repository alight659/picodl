# picodl Documentation

Full reference for every implemented module: what it does, how it works internally, and code examples.

---

# 1. Tensor (`tensor.py`)

The core autograd engine. Wraps a numpy array, tracks every operation performed on it, and computes gradients automatically via reverse-mode differentiation (backpropagation).

## Construction

```python
Tensor(data, requires_grad=False, _children=(), _op="")
```

- `data` - any array-like, converted to `np.float64` internally
- `requires_grad` - if True, gradients accumulate on `.backward()`
- `_children`, `_op` - internal, used to build the computation graph

### Attributes
- `.data` - underlying numpy array
- `.grad` - gradient array, `None` until `.backward()` is called
- `.shape` - property, returns `self.data.shape`
- `._prev` - set of parent Tensors in the graph
- `._backward` - closure that computes local gradient contribution

### Core methods
- `.zero_grad()` - sets `.grad = None`
- `.backward(grad=None)` - runs the backward pass

## How autograd works

Every op creates a new `Tensor` and attaches a `_backward` closure. `_backward` knows how to push gradient from the output back to its inputs. `backward()` does topological sort on the graph, then walks nodes in reverse order calling each `_backward()`.

```python
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
```

Gradient accumulation handles broadcasting automatically via `_accumulate`, which sum-reduces incoming gradient to match the original tensor's shape.

## Arithmetic ops

| Op | Usage | Notes |
|---|---|---|
| `+` | `x + y` | broadcasting supported |
| `-` | `x - y`, `-x` | via `__neg__` and `__sub__` |
| `*` | `x * y` | elementwise |
| `/` | `x / y` | implemented as `x * y**-1` |
| `**` | `x ** 2` | power, scalar exponent |
| `matmul` / `@` | `x.matmul(y)` or `x @ y` | matrix multiply |

### Example

```python
from picodl.tensor import Tensor

x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

z = (x * y).sum()
z.backward()

print(x.grad)  # [4. 5. 6.]  (dz/dx = y)
print(y.grad)  # [1. 2. 3.]  (dz/dy = x)
```

### Implementation detail: `__mul__`

```python
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
```

## Reduction ops

### `.sum(axis=None)`
Sums along axis (or all elements if `axis=None`). Gradient broadcasts back as ones.

```python
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
out = x.sum(axis=1)     # [3., 7.]
out.sum().backward()
print(x.grad)            # [[1. 1.] [1. 1.]]
```

### `.mean(axis=None)`
Same as sum but divides by count.

```python
x = Tensor([2.0, 4.0, 6.0], requires_grad=True)
m = x.mean()      # 4.0
m.backward()
print(x.grad)     # [0.333 0.333 0.333]
```

### `.max(axis=None, keepdims=False)`
Returns max values. Gradient routed only to the max position(s); ties split gradient evenly.

```python
x = Tensor([1.0, 5.0, 3.0], requires_grad=True)
m = x.max()
m.backward()
print(x.grad)   # [0. 1. 0.]
```

### `.argmax(axis=None)`
Plain numpy op, no gradient. Used at inference time (e.g. greedy decoding).

```python
x = Tensor([[0.1, 0.7, 0.2]])
x.argmax(axis=1)   # array([1])
```

## Activation functions

### `.tanh()`
```python
def tanh(self):
    t = np.tanh(self.data)
    out = Tensor(t, self.requires_grad, (self,), "tanh")

    def _backward():
        if self.requires_grad:
            self._accumulate(out.grad * (1 - t ** 2))
    out._backward = _backward
    return out
```

### `.relu()`
Gradient is 1 where input > 0, else 0.

```python
x = Tensor([-1.0, 2.0], requires_grad=True)
r = x.relu()          # [0. 2.]
r.sum().backward()
print(x.grad)          # [0. 1.]
```

### `.softmax(axis=-1)`
Numerically stable (subtracts max before exp). Full Jacobian-vector product in backward.

```python
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
```

```python
x = Tensor(np.random.randn(2, 3), requires_grad=True)
probs = x.softmax(axis=-1)
print(probs.data.sum(axis=-1))   # [1. 1.]
probs.sum().backward()
```

## Elementwise math

### `.log()`
```python
x = Tensor([1.0, np.e], requires_grad=True)
l = x.log()     # [0. 1.]
```

### `.sqrt()`
```python
x = Tensor([4.0], requires_grad=True)
s = x.sqrt()    # [2.]
s.backward()
print(x.grad)   # [0.25]  (d/dx sqrt(x) = 1/(2*sqrt(x)))
```

### `.clip(min_val, max_val)`
Gradient passes through only where value is within bounds (else 0).

```python
x = Tensor([0.5, 5.0, -1.0], requires_grad=True)
c = x.clip(0.0, 1.0)   # [0.5, 1.0, 0.0]
```

## Shape ops

### `.reshape(*shape)` / `.view(*shape)`
```python
x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
y = x.reshape(2, 12)
y.sum().backward()
print(x.grad.shape)   # (2, 3, 4)
```

### `.flatten(start_dim=0)`
Collapses all dims from `start_dim` onward into one.

```python
x = Tensor(np.random.randn(2, 3, 4, 4))
f = x.flatten(start_dim=1)
print(f.shape)   # (2, 48)
```

### `.transpose(*axes)` / `.permute(*axes)`
```python
x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
y = x.transpose(0, 2, 1)   # shape (2, 4, 3)
y.sum().backward()
```

## Indexing / gather-scatter ops

### `.__getitem__(idx)`
Standard numpy-style indexing, gradient scattered back via `np.add.at` (handles repeated indices correctly, important for embedding lookups).

```python
w = Tensor(np.arange(9.0).reshape(3, 3), requires_grad=True)
rows = w[[0, 2]]          # gather rows 0 and 2
rows.sum().backward()
print(w.grad[1])          # [0. 0. 0.]  -- row 1 untouched
```

### `.topk(k, axis=-1)`
Returns `(values_tensor, indices_array)`. Gradient scattered back to the original positions.

```python
x = Tensor([[1.0, 5.0, 3.0, 2.0]], requires_grad=True)
vals, idx = x.topk(2, axis=1)
print(vals.data)   # [[5. 3.]]
print(idx)         # [[1 2]]
```

### `.scatter(axis, index, src)`
Writes `src` values into a copy of self at `index` positions along `axis`. Used for MOE-style routing (write expert outputs into sparse slots).

```python
base = Tensor(np.zeros((3, 4)), requires_grad=True)
idx = np.array([[0], [1], [2]])
src = Tensor(np.ones((3, 1)), requires_grad=True)
out = base.scatter(1, idx, src)
```

### `.index_add(axis, index, src)`
Scatter-add: adds `src` rows into a copy of self at `index` positions (accumulates instead of overwrites). Used to combine sparse MOE expert outputs back into the full tensor.

```python
base = Tensor(np.zeros(3), requires_grad=True)
src = Tensor(np.array([1.0, 2.0]), requires_grad=True)
idx = np.array([0, 0])
out = base.index_add(0, idx, src)
print(out.data)   # [3. 0. 0.]
```

## Combine / split ops

### `Tensor.cat(tensors, axis=0)` (static method)
```python
a = Tensor(np.random.randn(2, 3), requires_grad=True)
b = Tensor(np.random.randn(2, 3), requires_grad=True)
c = Tensor.cat([a, b], axis=1)   # shape (2, 6)
```

### `Tensor.stack(tensors, axis=0)` (static method)
Adds a new axis, stacking tensors along it.

```python
a = Tensor(np.random.randn(3), requires_grad=True)
b = Tensor(np.random.randn(3), requires_grad=True)
s = Tensor.stack([a, b], axis=0)   # shape (2, 3)
```

### `.split(size_or_sections, axis=0)`
Inverse of `cat`. Accepts an int (equal chunks) or list of sizes.

```python
c = Tensor(np.random.randn(2, 6), requires_grad=True)
parts = c.split(3, axis=1)   # two tensors, each (2, 3)
```

## Sampling (no gradient, inference only)

### `.multinomial(num_samples=1)`
Samples class indices from a probability distribution per row.

```python
probs = Tensor([[0.0, 1.0, 0.0]])
probs.multinomial(num_samples=1)   # array([[1]])
```

## Convolution and pooling

### `.conv2d(kernel, stride=1, padding=0)`
Naive im2col-based 2D convolution. Input `(N, C, H, W)`, kernel `(out_c, C, kh, kw)`.

```python
x = Tensor(np.random.randn(1, 3, 8, 8), requires_grad=True)
kernel = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
out = x.conv2d(kernel, stride=1, padding=1)
print(out.shape)   # (1, 4, 8, 8)
out.sum().backward()
```

Implementation extracts sliding patches into a column matrix (`im2col`), reduces the convolution to a single matmul, then reshapes back. Backward reverses the process (`col2im`) to scatter gradient into overlapping input positions.

### `.maxpool2d(kernel_size, stride=None)`
Extracts patches same way as conv2d, takes max over each patch. Backward routes gradient only to the argmax position within each patch.

```python
x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
out = x.maxpool2d(2)     # shape (1, 2, 2, 2)
out.sum().backward()
```

### `.avgpool2d(kernel_size, stride=None)`
Same patch extraction, averages instead of max. Backward distributes gradient evenly across the patch.

```python
x = Tensor(np.ones((1, 2, 4, 4)))
out = x.avgpool2d(2)
print(out.data)   # all 1.0
```

---

# 2. Layers (`layers.py`)

All layers subclass `Layer`, which holds a `.params` dict of `Tensor(requires_grad=True)`.

```python
class Layer:
    def __init__(self):
        self.params: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError
```

No manual `.backward()` needed on layers - since everything is built from `Tensor` ops, autograd handles it automatically when you call `.backward()` on the final loss.

## Linear

```python
class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.params["w"] = Tensor(np.random.randn(input_size, output_size) * 0.1, requires_grad=True)
        self.params["b"] = Tensor(np.random.randn(output_size) * 0.1, requires_grad=True)

    def forward(self, inputs):
        return inputs.matmul(self.params["w"]) + self.params["b"]
```

```python
layer = Linear(4, 2)
x = Tensor(np.random.randn(5, 4))
out = layer.forward(x)      # shape (5, 2)
```

## Tanh / ReLU

Stateless activations, no params.

```python
out = Tanh().forward(x)
out = ReLU().forward(x)
```

## Conv2D

```python
Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

Weight shape `(out_channels, in_channels, kernel_size, kernel_size)`, bias shape `(out_channels,)`. Bias is reshaped to `(1, out_c, 1, 1)` to broadcast correctly across `(N, C, H, W)`.

```python
conv = Conv2D(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
x = Tensor(np.random.randn(2, 3, 16, 16), requires_grad=True)
out = conv.forward(x)       # (2, 8, 16, 16)
out.sum().backward()
```

## Embedding

```python
class Embedding(Layer):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.params["w"] = Tensor(np.random.randn(vocab_size, embed_dim) * 0.1, requires_grad=True)

    def forward(self, idx):
        return self.params["w"][idx]
```

Input `idx` is a **plain python list or numpy int array**, not a Tensor.

```python
emb = Embedding(vocab_size=1000, embed_dim=32)
idx = [1, 5, 9]
out = emb.forward(idx)      # shape (3, 32)
out.sum().backward()        # grad only flows into rows 1, 5, 9
```

## LayerNorm

Normalizes over the last axis: `(x - mean) / sqrt(var + eps) * gamma + beta`.

```python
ln = LayerNorm(dim=16)
x = Tensor(np.random.randn(4, 16), requires_grad=True)
out = ln.forward(x)
out.sum().backward()
```

Gradient flows fully through mean/var since they're computed via `Tensor.mean()` inside forward, not raw numpy.

## MaxPool2D / AvgPool2D

```python
MaxPool2D(kernel_size, stride=None)
AvgPool2D(kernel_size, stride=None)
```

Thin wrappers around `Tensor.maxpool2d` / `Tensor.avgpool2d`.

```python
x = Tensor(np.random.randn(1, 3, 8, 8), requires_grad=True)
out = MaxPool2D(2).forward(x)    # (1, 3, 4, 4)
```

## Flatten

```python
class Flatten(Layer):
    def forward(self, inputs):
        return inputs.flatten(start_dim=1)
```

```python
x = Tensor(np.random.randn(2, 3, 4, 4))
out = Flatten().forward(x)   # (2, 48)
```

Typical CNN tail: `Conv -> Pool -> Flatten -> Linear`.

## BatchNorm2D

Normalizes over `(N, H, W)` per channel. Tracks running mean/var for inference (`.eval()` mode), like standard batchnorm.

```python
bn = BatchNorm2D(num_features=3)
x = Tensor(np.random.randn(4, 3, 8, 8), requires_grad=True)

out = bn.forward(x)      # training mode: uses batch stats, updates running stats
out.sum().backward()

bn.eval()
out_inference = bn.forward(x)   # uses running_mean / running_var instead
bn.train()                       # switch back
```

**Note:** batch mean/var are computed directly from `.data` (treated as constants during backward), a common simplification. Gradient still flows correctly through `gamma`/`beta` and through `(x - mean)/std`, just not through the mean/var statistics themselves.

## Dropout

```python
Dropout(p=0.5)
```

Randomly zeroes elements with probability `p` during training, scales survivors by `1/(1-p)` (inverted dropout). Identity during `.eval()`.

```python
do = Dropout(p=0.5)
out_train = do.forward(x)    # random mask applied

do.eval()
out_eval = do.forward(x)     # identity, out_eval == x
do.train()
```

---

# 3. Loss functions (`loss.py`)

All return a scalar `Tensor`; call `.backward()` on the result to backprop through the whole network.

## MSE

```python
class MSE(Loss):
    def loss(self, predicted, actual):
        diff = predicted - actual
        return (diff * diff).sum()
```

```python
pred = Tensor([1.0, 2.0], requires_grad=True)
actual = Tensor([0.0, 0.0])
loss = MSE().loss(pred, actual)
loss.backward()
print(pred.grad)   # [2. 4.]
```

## BinaryCrossEntropy

```python
class BinaryCrossEntropy(Loss):
    def loss(self, predicted, actual):
        eps = 1e-9
        p = predicted.clip(eps, 1 - eps)
        term1 = actual * p.log()
        term2 = (1 - actual) * (1 - p).log()
        return -(term1 + term2).sum()
```

Predicted values are clipped to `[eps, 1-eps]` to avoid `log(0)`.

```python
pred = Tensor([0.9, 0.1], requires_grad=True)
actual = Tensor([1.0, 0.0])
loss = BinaryCrossEntropy().loss(pred, actual)
loss.backward()
```

## NLLLoss

Negative log-likelihood. `predicted` is a log-probability Tensor `(N, C)`, `actual` is a **plain int array** of class indices `(N,)`, not a Tensor.

```python
class NLLLoss(Loss):
    def loss(self, predicted, actual):
        n = predicted.shape[0]
        picked = predicted[np.arange(n), actual]
        return -picked.mean()
```

```python
log_probs = Tensor(np.log([[0.7, 0.3], [0.2, 0.8]]), requires_grad=True)
targets = np.array([0, 1])
loss = NLLLoss().loss(log_probs, targets)
loss.backward()
```

For a full cross-entropy setup, combine `.softmax().log()` (or a fused log-softmax) with `NLLLoss`.

---

# 4. Optimizers (`optim.py`)

All optimizers implement `.step(net)`, iterating `net.params()` and updating `.data` in place using `.grad`.

## SGD

```python
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, net):
        for param in net.params():
            if param.grad is not None:
                param.data -= self.lr * param.grad
```

```python
optimizer = SGD(lr=0.01)
optimizer.step(net)
```

## RMSprop

Keeps a running average of squared gradients per parameter, divides update by its square root.

```python
class RMSprop(Optimizer):
    def __init__(self, lr=0.001, decay=0.9, eps=1e-8):
        ...
    def step(self, net):
        for param in net.params():
            key = id(param)
            self.cache[key] = self.decay * self.cache[key] + (1 - self.decay) * (param.grad ** 2)
            param.data -= self.lr * param.grad / (np.sqrt(self.cache[key]) + self.eps)
```

```python
optimizer = RMSprop(lr=0.001, decay=0.9)
optimizer.step(net)
```

## Adam

Momentum (`m`) + adaptive scaling (`v`), both bias-corrected using timestep `t`.

```python
class Adam(Optimizer):
    def step(self, net):
        self.t += 1
        for param in net.params():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * param.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

```python
optimizer = Adam(lr=0.001, beta1=0.9, beta2=0.999)
for step in range(100):
    net.zero_grad()
    out = net.forward(x)
    loss = MSE().loss(out, target)
    loss.backward()
    optimizer.step(net)
```

Each optimizer tracks its own per-parameter state keyed by `id(param)`, so switching optimizers mid-training starts fresh state.

---

# 5. NeuralNet (`nn.py`)

Container that chains layers and exposes params/serialization.

```python
class NeuralNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def params(self):
        for layer in self.layers:
            for param in layer.params.values():
                yield param

    def zero_grad(self):
        for param in self.params():
            param.zero_grad()
```

```python
net = NeuralNet([
    Linear(2, 8),
    Tanh(),
    Linear(8, 1),
])
out = net.forward(Tensor(np.random.randn(10, 2)))
```

## Serialization

```python
def save(self, path):
    flat = {}
    for i, layer in enumerate(self.layers):
        for name, param in layer.params.items():
            flat[f"{i}_{name}"] = param.data
    np.savez(path, **flat)

def load(self, path):
    data = np.load(path)
    for i, layer in enumerate(self.layers):
        for name, param in layer.params.items():
            key = f"{i}_{name}"
            if key in data:
                param.data = data[key]
            else:
                raise KeyError(f"missing {key} in {path}, architecture mismatch")
```

Params are keyed by `{layer_index}_{param_name}`, stored in a single `.npz` file.

```python
net.save("model.npz")

net2 = NeuralNet([Linear(2, 8), Tanh(), Linear(8, 1)])  # same architecture first
net2.load("model.npz")
```

**Important:** `load()` requires the target net to already have the same architecture built (same layer count, order, and param names/shapes). Only weight *values* are restored, not structure.

---

# 6. Data (`data.py`)

## BatchIterator

```python
class BatchIterator(DataIterator):
    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        for start in starts:
            end = start + self.batch_size
            yield Batch(inputs[start:end], targets[start:end])
```

Takes **raw numpy arrays** (not Tensors), yields `Batch(inputs, targets)` namedtuples.

```python
inputs = np.random.randn(100, 2)
targets = np.random.randn(100, 1)

for batch in BatchIterator(batch_size=16, shuffle=True)(inputs, targets):
    print(batch.inputs.shape, batch.targets.shape)
```

---

# 7. Train loop (`train.py`)

```python
def train(net, inputs, targets, num_epochs=1000, iterator=BatchIterator(), loss=MSE(), optimizer=SGD()):
    raw_inputs = inputs.data if isinstance(inputs, Tensor) else inputs
    raw_targets = targets.data if isinstance(targets, Tensor) else targets
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch in iterator(raw_inputs, raw_targets):
            net.zero_grad()
            batch_in = Tensor(batch.inputs)
            # classification losses (CrossEntropyLoss, NLLLoss) expect plain int
            # class-index targets, not a Tensor - only wrap float targets
            targets_are_int = np.issubdtype(np.asarray(batch.targets).dtype, np.integer)
            batch_tgt = batch.targets if targets_are_int else Tensor(batch.targets)
            predicted = net.forward(batch_in)
            loss_val = loss.loss(predicted, batch_tgt)
            epoch_loss += loss_val.data
            loss_val.backward()
            optimizer.step(net)
        print(epoch, epoch_loss)
```

Accepts either a `Tensor` or raw numpy array for `inputs`/`targets` (unwraps `.data` internally so `BatchIterator` always gets raw arrays).

**Target handling:** `train()` checks the dtype of `batch.targets` each batch. Integer-dtype targets (class indices, for `CrossEntropyLoss`/`NLLLoss`) are passed through raw; anything else (float targets, for `MSE`/`BinaryCrossEntropy`) gets wrapped in a `Tensor`. This matters because `CrossEntropyLoss`/`NLLLoss` do numpy-style fancy indexing on the target array internally (`predicted[np.arange(n), actual]`) - passing a `Tensor` there instead of a raw int array would break that indexing.

```python
# classification: y_train is an int array, e.g. np.array([3, 0, 7, ...])
train(net, X_train, y_train, loss=CrossEntropyLoss(), optimizer=AdamW())

# regression: y_train is a float array, gets wrapped in Tensor automatically
train(net, X_train, y_train, loss=MSE(), optimizer=Adam())
```

Per batch:
1. `net.zero_grad()` - clear previous gradients
2. forward pass through the net
3. compute scalar loss
4. `loss.backward()` - populate `.grad` on every param
5. `optimizer.step(net)` - update params using their `.grad`

## Full example

```python
from picodl.nn import NeuralNet
from picodl.layers import Linear, Tanh
from picodl.loss import MSE
from picodl.optim import Adam
from picodl.train import train
from picodl.tensor import Tensor
import numpy as np

net = NeuralNet([
    Linear(2, 8),
    Tanh(),
    Linear(8, 1),
])

inputs = Tensor(np.random.randn(100, 2))
targets = Tensor(np.random.randn(100, 1))

train(net, inputs, targets, num_epochs=50, loss=MSE(), optimizer=Adam(lr=0.01))

net.save("model.npz")
```

## CNN example

```python
from picodl.nn import NeuralNet
from picodl.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear, BatchNorm2D

net = NeuralNet([
    Conv2D(1, 8, kernel_size=3, padding=1),
    BatchNorm2D(8),
    ReLU(),
    MaxPool2D(2),
    Flatten(),
    Linear(8 * 14 * 14, 10),
])

x = Tensor(np.random.randn(4, 1, 28, 28))  # e.g. MNIST-style batch
out = net.forward(x)   # (4, 10)
```

---

# Roadmap (not yet implemented)

- GPU support (cupy backend swap)
- Distributed training (multi-process gradient sync)

---

# 8. Newly added (this section)

Detailed reference for `log_softmax`, `gelu`, `conv2d` padding modes, `GELU`, `GlobalAvgPool`, `TiedLinear`, `CrossEntropyLoss`, `AdamW`.

## 8.1 `Tensor.log_softmax(axis=-1)`

Computes `log(softmax(x))` directly in one numerically stable pass, instead of chaining `.softmax().log()` (which can lose precision when probabilities get close to 0).

### Why not just `x.softmax().log()`?

`softmax` can underflow to exactly `0.0` for very negative logits, and `log(0)` is `-inf` / `nan`. `log_softmax` sidesteps this using the log-sum-exp trick:

```
log_softmax(x)_i = x_i - max(x) - log( sum_j exp(x_j - max(x)) )
```

No exponential of the full range ever needs to be small enough to underflow before the subtraction happens.

### Implementation

```python
def log_softmax(self, axis=-1):
    shifted = self.data - np.max(self.data, axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    result = shifted - logsumexp
    out = Tensor(result, self.requires_grad, (self,), "log_softmax")

    def _backward():
        if self.requires_grad:
            s = np.exp(result)  # recover softmax probs cheaply
            grad_out = out.grad
            sum_grad = grad_out.sum(axis=axis, keepdims=True)
            self._accumulate(grad_out - s * sum_grad)
    out._backward = _backward
    return out
```

The backward formula `grad_out - softmax(x) * sum(grad_out)` is the standard Jacobian-vector product for log-softmax, cheaper to compute than differentiating through softmax and log separately.

### Example

```python
from picodl.tensor import Tensor
import numpy as np

logits = Tensor(np.random.randn(2, 5), requires_grad=True)
log_probs = logits.log_softmax(axis=-1)

print(np.exp(log_probs.data).sum(axis=-1))  # [1. 1.]  -- exp(log_softmax) sums to 1

log_probs.sum().backward()
print(logits.grad.shape)  # (2, 5)
```

---

## 8.2 `Tensor.gelu()`

Gaussian Error Linear Unit activation, using the standard tanh approximation (same one used in GPT-2/BERT-style implementations):

```
gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715*x^3) ))
```

Smooth alternative to ReLU - doesn't have a hard cutoff at 0, so gradient doesn't die for slightly negative inputs.

### Implementation

```python
def gelu(self):
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
```

Backward is the product rule applied to `0.5*x*(1+tanh(inner(x)))` - `dresult` combines the derivative of the outer `x * (...)` term and the chain rule through `tanh(inner(x))`.

### Layer wrapper

```python
class GELU(Layer):
    def forward(self, inputs):
        return inputs.gelu()
```

### Example

```python
from picodl.tensor import Tensor
from picodl.layers import GELU

x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
out = GELU().forward(x)
print(out.data)   # smoothly curves through 0, unlike ReLU's hard kink

out.sum().backward()
print(x.grad)      # nonzero even for negative x, unlike ReLU
```

---

## 8.3 Convolution padding modes

`Tensor.conv2d` and `Conv2D` layer now accept a `pad_mode` argument.

| `pad_mode` | numpy equivalent | Behavior |
|---|---|---|
| `"constant"` (default) | `mode="constant"` | zero-pad, standard, **exact gradient** |
| `"reflect"` | `mode="reflect"` | mirrors interior values across the edge |
| `"replicate"` | `mode="edge"` | repeats the edge pixel outward |
| `"circular"` | `mode="wrap"` | wraps around, like a torus |

### Implementation

```python
def conv2d(self, kernel, stride=1, padding=0, pad_mode="constant"):
    ...
    np_mode = {"constant": "constant", "reflect": "reflect",
               "replicate": "edge", "circular": "wrap"}[pad_mode]
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)), mode=np_mode)
    ...
```

### Important caveat

Backward pass is implemented as a **crop** (take the gradient of the padded region and just slice it back off). This is mathematically exact for `"constant"` padding (padded values are constants, contribute no gradient to the original tensor). For `"reflect"`, `"replicate"`, `"circular"`, the padded values are literally *copies* of interior values, so a fully correct backward would need to **add** the cropped-off gradient back onto the positions it was copied from. This implementation does **not** do that scatter-add step - it's an approximation. Good enough for most training runs where padding is just a couple pixels, but be aware of it if you see gradient checks fail near the border with non-constant modes.

### Example

```python
from picodl.tensor import Tensor
from picodl.layers import Conv2D
import numpy as np

x = Tensor(np.random.randn(1, 3, 8, 8), requires_grad=True)

conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, padding=1, pad_mode="reflect")
out = conv.forward(x)
print(out.shape)  # (1, 4, 8, 8) -- same spatial size due to padding=1 with 3x3 kernel

out.sum().backward()
```

---

## 8.4 `GlobalAvgPool`

Averages every spatial position down to a single value per channel: `(N, C, H, W) -> (N, C)`. Common at the end of a CNN backbone right before a classifier head, as a lighter-weight alternative to `Flatten` + big `Linear` (avoids a huge parameter count from flattening a large spatial map).

### Implementation

```python
class GlobalAvgPool(Layer):
    def forward(self, inputs):
        return inputs.mean(axis=3).mean(axis=2)
```

Reduces `W` first (`axis=3`), then `H` (`axis=2` on the now-3D result) — equivalent to averaging over both spatial axes at once, reusing the existing `.mean()` op so gradient (uniform 1/N spread) flows for free.

### Example

```python
from picodl.tensor import Tensor
from picodl.layers import Conv2D, ReLU, GlobalAvgPool, Linear
from picodl.nn import NeuralNet
import numpy as np

net = NeuralNet([
    Conv2D(1, 16, kernel_size=3, padding=1),
    ReLU(),
    GlobalAvgPool(),     # (N, 16, H, W) -> (N, 16)
    Linear(16, 10),       # classifier head, tiny compared to Flatten's H*W*16 input
])

x = Tensor(np.random.randn(4, 1, 28, 28))
out = net.forward(x)
print(out.shape)  # (4, 10)
```

---

## 8.5 `TiedLinear` - weight tying

Shares a weight `Tensor` between two layers instead of allocating a second one. The classic use case: a language model's input token embedding and its output projection (`lm_head`) often use the **same** matrix (transposed), which both saves parameters and tends to improve quality.

### Implementation

```python
class TiedLinear(Layer):
    def __init__(self, tied_layer, weight_key="w", bias=False):
        super().__init__()
        self.tied_weight = tied_layer.params[weight_key]  # SAME Tensor object, not a copy
        if bias:
            out_features = self.tied_weight.data.shape[0]
            self.params["b"] = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, inputs):
        out = inputs.matmul(self.tied_weight.transpose(1, 0))
        if "b" in self.params:
            out = out + self.params["b"]
        return out
```

### How the sharing actually works

`self.tied_weight` is a **reference** to the exact same `Tensor` object stored in the embedding layer's `params["w"]` - not a copy of its data. Because of that:

- Every `.forward()` call builds a fresh graph node using `.transpose()`, but that node's `_prev` points back to the *same* underlying param.
- When you call `.backward()`, gradient contributions from **both** uses (the embedding lookup *and* the tied linear projection) accumulate into that one shared `Tensor.grad`, since `_accumulate` just adds into whatever `.grad` already holds.
- `TiedLinear.params` is **empty** for the weight (it's not re-registered), so `NeuralNet.params()` only yields it once - the optimizer updates it exactly once per step, not twice.

### Example

```python
from picodl.tensor import Tensor
from picodl.layers import Embedding, TiedLinear
import numpy as np

vocab_size, embed_dim = 100, 16
embedding = Embedding(vocab_size, embed_dim)
lm_head = TiedLinear(embedding)   # shares embedding.params["w"], shape (100, 16)

idx = [3, 7, 12]
hidden = embedding.forward(idx)          # (3, 16)
logits = lm_head.forward(hidden)          # (3, 100) -- reuses embedding weight transposed

loss = logits.sum()
loss.backward()

# gradient landed on the ONE shared tensor:
print(embedding.params["w"].grad.shape)   # (100, 16)
```

### Using it inside a NeuralNet

```python
from picodl.nn import NeuralNet

# NOTE: NeuralNet.params() will only see embedding's "w" once, since
# TiedLinear doesn't register its own copy - safe for optimizer.step()
net = NeuralNet([embedding, lm_head])
```

---

## 8.6 `CrossEntropyLoss`

Combines `log_softmax` and `NLLLoss` into a single step, matching the standard "cross entropy from raw logits" pattern used almost everywhere (PyTorch's `nn.CrossEntropyLoss` works the same way).

### Why combine them?

Doing `predicted.softmax().log()` then picking target log-probs is both less numerically stable (see 8.1) and slightly slower (two backward passes fused into one gives a simpler combined gradient). `CrossEntropyLoss` takes **raw, un-normalized logits** directly - you should *not* apply softmax yourself before calling it.

### Implementation

```python
class CrossEntropyLoss(Loss):
    """predicted = raw logits (N, C), actual = int class indices (N,)."""
    def loss(self, predicted, actual):
        log_probs = predicted.log_softmax(axis=-1)
        n = predicted.shape[0]
        picked = log_probs[np.arange(n), actual]
        return -picked.mean()
```

- `predicted.log_softmax(axis=-1)` - stable log-probabilities per class
- `log_probs[np.arange(n), actual]` - fancy-index gather, pulls out the log-prob of the *correct* class for each row (reuses `Tensor.__getitem__`, so gradient scatters back correctly through log_softmax's own backward)
- `-picked.mean()` - average negative log-likelihood across the batch

### Example

```python
from picodl.tensor import Tensor
from picodl.loss import CrossEntropyLoss
import numpy as np

logits = Tensor(np.random.randn(8, 10), requires_grad=True)  # 8 examples, 10 classes
targets = np.array([3, 0, 7, 7, 1, 9, 2, 5])                  # plain int array, NOT a Tensor

loss_fn = CrossEntropyLoss()
loss = loss_fn.loss(logits, targets)
print(loss.data)  # scalar

loss.backward()
print(logits.grad.shape)  # (8, 10)
```

### Difference vs `NLLLoss`

| | Input `predicted` | Extra step needed |
|---|---|---|
| `NLLLoss` | already log-probabilities | you must call `.log_softmax()` yourself first |
| `CrossEntropyLoss` | raw logits | none - does log_softmax internally |

```python
from picodl.loss import NLLLoss, CrossEntropyLoss

# these two produce the same result:
loss_a = CrossEntropyLoss().loss(logits, targets)
loss_b = NLLLoss().loss(logits.log_softmax(axis=-1), targets)
```

---

## 8.7 `AdamW`

Adam with **decoupled weight decay**, the optimizer almost every transformer is trained with (as opposed to plain Adam with L2 regularization folded into the gradient, which behaves differently once you have adaptive per-parameter learning rates).

### Why "decoupled" matters

Plain "Adam + L2" adds `weight_decay * param` directly onto the gradient before it goes through the momentum/variance running averages. That means the decay term itself gets scaled and reshaped by Adam's adaptive learning rate machinery - not what you'd expect from "decay". AdamW instead applies the decay as a **separate, plain multiplicative shrink** on the weights, independent of the gradient statistics.

### Implementation

```python
class AdamW(Optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.weight_decay = weight_decay
        self.m, self.v, self.t = {}, {}, 0

    def step(self, net):
        self.t += 1
        for param in net.params():
            if param.grad is None:
                continue
            key = id(param)
            if key not in self.m:
                self.m[key] = np.zeros_like(param.data)
                self.v[key] = np.zeros_like(param.data)

            # decoupled weight decay - independent of gradient/momentum
            param.data -= self.lr * self.weight_decay * param.data

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * param.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

Two separate subtractions from `param.data` each step:
1. the weight-decay shrink (`lr * weight_decay * param.data`)
2. the usual bias-corrected Adam update (`lr * m_hat / (sqrt(v_hat) + eps)`)

State (`m`, `v`) is keyed by `id(param)`, same pattern as plain `Adam` - each parameter tracks its own momentum/variance history independent of other parameters.

### Example

```python
from picodl.nn import NeuralNet
from picodl.layers import Linear, GELU
from picodl.loss import MSE
from picodl.optim import AdamW
from picodl.tensor import Tensor
import numpy as np

net = NeuralNet([Linear(4, 16), GELU(), Linear(16, 1)])
optimizer = AdamW(lr=0.001, weight_decay=0.01)

x = Tensor(np.random.randn(32, 4))
y = Tensor(np.random.randn(32, 1))

for step in range(100):
    net.zero_grad()
    out = net.forward(x)
    loss = MSE().loss(out, y)
    loss.backward()
    optimizer.step(net)
```

### When to prefer AdamW over Adam

Use `AdamW` whenever you want weight decay as a regularizer (common for transformer-style models). Use plain `SGD`/`Adam` (no `weight_decay`) for small toy nets where regularization isn't the concern.