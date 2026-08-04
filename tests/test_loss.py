import numpy as np
from picodl.tensor import Tensor
from picodl.loss import MSE, BinaryCrossEntropy, NLLLoss, CrossEntropyLoss


def test_mse_zero_when_equal():
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    actual = Tensor([1.0, 2.0, 3.0])
    loss = MSE().loss(pred, actual)
    assert np.isclose(loss.data, 0.0)


def test_mse_backward():
    pred = Tensor([1.0, 2.0], requires_grad=True)
    actual = Tensor([0.0, 0.0])
    loss = MSE().loss(pred, actual)
    loss.backward()
    assert np.allclose(pred.grad, [2.0, 4.0])


def test_bce_reasonable_range():
    pred = Tensor([0.9, 0.1], requires_grad=True)
    actual = Tensor([1.0, 0.0])
    loss = BinaryCrossEntropy().loss(pred, actual)
    assert loss.data > 0
    loss.backward()
    assert pred.grad is not None


def test_bce_clip_no_nan():
    pred = Tensor([0.0, 1.0], requires_grad=True)
    actual = Tensor([1.0, 0.0])
    loss = BinaryCrossEntropy().loss(pred, actual)
    assert not np.isnan(loss.data)
    loss.backward()
    assert not np.any(np.isnan(pred.grad))


def test_nll_loss():
    log_probs = Tensor(np.log(np.array([[0.7, 0.3], [0.2, 0.8]])), requires_grad=True)
    targets = np.array([0, 1])
    loss = NLLLoss().loss(log_probs, targets)
    expected = -np.mean([np.log(0.7), np.log(0.8)])
    assert np.isclose(loss.data, expected)
    loss.backward()
    assert log_probs.grad is not None


def test_cross_entropy_loss():
    logits = Tensor(np.random.randn(8, 10), requires_grad=True)
    targets = np.array([3, 0, 7, 7, 1, 9, 2, 5])
    loss = CrossEntropyLoss().loss(logits, targets)
    loss.backward()
    assert logits.grad.shape == (8, 10)


def test_cross_entropy_matches_manual_log_softmax_nll():
    logits = Tensor(np.random.randn(4, 6), requires_grad=True)
    targets = np.array([0, 1, 2, 3])
    ce = CrossEntropyLoss().loss(logits, targets)

    logits2 = Tensor(logits.data.copy(), requires_grad=True)
    manual = NLLLoss().loss(logits2.log_softmax(axis=-1), targets)
    assert np.isclose(ce.data, manual.data)
