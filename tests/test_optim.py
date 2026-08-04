import numpy as np
from picodl.tensor import Tensor
from picodl.nn import NeuralNet
from picodl.layers import Linear
from picodl.optim import SGD, RMSprop, Adam, AdamW


def make_net():
    return NeuralNet([Linear(3, 2)])


def run_one_step(optimizer):
    net = make_net()
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    target = Tensor(np.random.randn(4, 2))
    out = net.forward(x)
    loss = ((out - target) * (out - target)).sum()
    net.zero_grad()
    loss.backward()
    before = {id(p): p.data.copy() for p in net.params()}
    optimizer.step(net)
    after = {id(p): p.data for p in net.params()}
    changed = any(not np.allclose(before[k], after[k]) for k in before)
    return changed


def test_sgd_updates_params():
    assert run_one_step(SGD(lr=0.1))


def test_rmsprop_updates_params():
    assert run_one_step(RMSprop(lr=0.01))


def test_adam_updates_params():
    assert run_one_step(Adam(lr=0.01))


def test_adamw_updates_params():
    assert run_one_step(AdamW(lr=0.01, weight_decay=0.01))


def test_adam_bias_correction_step_count():
    net = make_net()
    optimizer = Adam(lr=0.01)
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    target = Tensor(np.random.randn(2, 2))
    for _ in range(3):
        net.zero_grad()
        out = net.forward(x)
        loss = ((out - target) * (out - target)).sum()
        loss.backward()
        optimizer.step(net)
    assert optimizer.t == 3


def test_adamw_weight_decay_shrinks_even_with_zero_grad():
    net = make_net()
    optimizer = AdamW(lr=0.1, weight_decay=0.5)
    p = next(net.params())
    p.grad = np.zeros_like(p.data)
    before = p.data.copy()
    optimizer.step(net)
    assert np.all(np.abs(p.data) <= np.abs(before) + 1e-12)


def test_sgd_no_grad_no_crash():
    net = make_net()
    optimizer = SGD(lr=0.1)
    optimizer.step(net)
