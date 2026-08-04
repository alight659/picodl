import os
import numpy as np
from picodl.tensor import Tensor
from picodl.layers import GELU, GlobalAvgPool, Embedding, TiedLinear, Conv2D, Linear
from picodl.loss import CrossEntropyLoss, NLLLoss
from picodl.optim import AdamW
from picodl.nn import NeuralNet


def test_log_softmax_sums_to_one():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    log_probs = x.log_softmax(axis=-1)
    assert np.allclose(np.exp(log_probs.data).sum(axis=-1), 1.0)


def test_log_softmax_backward_shape():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    log_probs = x.log_softmax(axis=-1)
    log_probs.sum().backward()
    assert x.grad.shape == (3, 4)


def test_log_softmax_matches_manual_softmax_log():
    x = Tensor(np.random.randn(2, 5))
    ls = x.log_softmax(axis=-1)
    manual = np.log(np.exp(x.data - x.data.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True))
    expected = (x.data - x.data.max(axis=-1, keepdims=True)) - manual
    assert np.allclose(ls.data, expected)


def test_gelu_zero_at_zero():
    x = Tensor([0.0], requires_grad=True)
    out = x.gelu()
    assert np.isclose(out.data[0], 0.0)


def test_gelu_backward_nonzero_for_negative():
    x = Tensor([-1.0], requires_grad=True)
    out = x.gelu()
    out.backward()
    assert x.grad[0] != 0  # unlike relu, gradient survives for negative input


def test_gelu_layer():
    x = Tensor(np.random.randn(4, 4), requires_grad=True)
    out = GELU().forward(x)
    assert out.shape == (4, 4)
    out.sum().backward()
    assert x.grad is not None


def test_conv2d_constant_padding_default():
    x = Tensor(np.random.randn(1, 2, 6, 6), requires_grad=True)
    kernel = Tensor(np.random.randn(3, 2, 3, 3), requires_grad=True)
    out = x.conv2d(kernel, padding=1)
    assert out.shape == (1, 3, 6, 6)
    out.sum().backward()
    assert x.grad.shape == (1, 2, 6, 6)


def test_conv2d_reflect_padding_shape():
    x = Tensor(np.random.randn(1, 2, 6, 6), requires_grad=True)
    kernel = Tensor(np.random.randn(3, 2, 3, 3), requires_grad=True)
    out = x.conv2d(kernel, padding=1, pad_mode="reflect")
    assert out.shape == (1, 3, 6, 6)
    out.sum().backward()
    assert x.grad.shape == (1, 2, 6, 6)


def test_conv2d_replicate_and_circular_padding_run():
    x = Tensor(np.random.randn(1, 1, 5, 5), requires_grad=True)
    kernel = Tensor(np.random.randn(1, 1, 3, 3), requires_grad=True)
    for mode in ("replicate", "circular"):
        out = x.conv2d(kernel, padding=1, pad_mode=mode)
        assert out.shape == (1, 1, 5, 5)


def test_conv2d_layer_pad_mode_param():
    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1, pad_mode="reflect")
    x = Tensor(np.random.randn(1, 2, 6, 6), requires_grad=True)
    out = conv.forward(x)
    assert out.shape == (1, 3, 6, 6)
    out.sum().backward()
    assert conv.params["w"].grad is not None


def test_global_avg_pool_shape():
    x = Tensor(np.random.randn(2, 3, 5, 5), requires_grad=True)
    out = GlobalAvgPool().forward(x)
    assert out.shape == (2, 3)


def test_global_avg_pool_value():
    x = Tensor(np.ones((1, 2, 4, 4)))
    out = GlobalAvgPool().forward(x)
    assert np.allclose(out.data, 1.0)


def test_global_avg_pool_backward():
    x = Tensor(np.random.randn(2, 3, 4, 4), requires_grad=True)
    out = GlobalAvgPool().forward(x)
    out.sum().backward()
    assert x.grad.shape == (2, 3, 4, 4)
    assert np.allclose(x.grad, 1.0 / 16.0)


def test_tied_linear_shares_gradient():
    emb = Embedding(vocab_size=10, embed_dim=4)
    tied = TiedLinear(emb)
    idx = [1, 2]
    hidden = emb.forward(idx)
    logits = tied.forward(hidden)
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert emb.params["w"].grad is not None


def test_tied_linear_not_double_registered_in_net_params():
    emb = Embedding(vocab_size=5, embed_dim=3)
    tied = TiedLinear(emb)
    net = NeuralNet([emb, tied])
    params = list(net.params())
    # only emb's "w" should appear once, tied contributes nothing extra (no bias)
    assert len(params) == 1


def test_tied_linear_with_bias():
    emb = Embedding(vocab_size=6, embed_dim=4)
    tied = TiedLinear(emb, bias=True)
    assert "b" in tied.params
    assert tied.params["b"].shape == (6,)
    idx = [0, 1]
    hidden = emb.forward(idx)
    out = tied.forward(hidden)
    assert out.shape == (2, 6)


def test_cross_entropy_matches_manual_log_softmax_nll():
    logits = Tensor(np.random.randn(4, 6), requires_grad=True)
    targets = np.array([0, 1, 2, 3])
    ce = CrossEntropyLoss().loss(logits, targets)

    logits2 = Tensor(logits.data.copy(), requires_grad=True)
    manual = NLLLoss().loss(logits2.log_softmax(axis=-1), targets)

    assert np.isclose(ce.data, manual.data)


def test_cross_entropy_backward():
    logits = Tensor(np.random.randn(5, 3), requires_grad=True)
    targets = np.array([0, 2, 1, 1, 0])
    loss = CrossEntropyLoss().loss(logits, targets)
    loss.backward()
    assert logits.grad.shape == (5, 3)


def test_cross_entropy_lower_when_confident_and_correct():
    confident_correct = Tensor(np.array([[10.0, 0.0, 0.0]]), requires_grad=True)
    unsure = Tensor(np.array([[0.1, 0.0, 0.0]]), requires_grad=True)
    targets = np.array([0])
    loss_confident = CrossEntropyLoss().loss(confident_correct, targets)
    loss_unsure = CrossEntropyLoss().loss(unsure, targets)
    assert loss_confident.data < loss_unsure.data


def test_adamw_updates_params():
    net = NeuralNet([Linear(3, 2)])
    optimizer = AdamW(lr=0.01, weight_decay=0.01)
    x = Tensor(np.random.randn(4, 3), requires_grad=True)
    target = Tensor(np.random.randn(4, 2))
    out = net.forward(x)
    loss = ((out - target) * (out - target)).sum()
    net.zero_grad()
    loss.backward()
    before = {id(p): p.data.copy() for p in net.params()}
    optimizer.step(net)
    after = {id(p): p.data for p in net.params()}
    assert any(not np.allclose(before[k], after[k]) for k in before)


def test_adamw_weight_decay_shrinks_even_with_zero_grad():
    param_holder = NeuralNet([Linear(2, 2)])
    optimizer = AdamW(lr=0.1, weight_decay=0.5)
    p = next(param_holder.params())
    p.grad = np.zeros_like(p.data)  # zero gradient
    before = p.data.copy()
    optimizer.step(param_holder)
    # weight decay term alone should shrink the weights toward zero
    assert np.all(np.abs(p.data) <= np.abs(before) + 1e-12)


def test_adamw_step_counter_increments():
    net = NeuralNet([Linear(2, 2)])
    optimizer = AdamW(lr=0.01)
    x = Tensor(np.random.randn(3, 2), requires_grad=True)
    for i in range(4):
        net.zero_grad()
        out = net.forward(x)
        out.sum().backward()
        optimizer.step(net)
    assert optimizer.t == 4


def test_save_load_custom_extension(tmp_path):
    net = NeuralNet([Linear(2, 2)])
    path = str(tmp_path / "model.pdl")
    net.save(path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".npz")  # no forced extension appended

    net2 = NeuralNet([Linear(2, 2)])
    net2.load(path)
    for p1, p2 in zip(net.params(), net2.params()):
        assert np.allclose(p1.data, p2.data)
