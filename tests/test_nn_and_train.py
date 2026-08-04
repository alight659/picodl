import os
import numpy as np
from picodl.tensor import Tensor
from picodl.nn import NeuralNet
from picodl.layers import Linear, Tanh, GELU
from picodl.loss import MSE, CrossEntropyLoss
from picodl.optim import SGD, AdamW
from picodl.data import BatchIterator
from picodl.train import train


def test_neuralnet_forward_chain():
    net = NeuralNet([Linear(2, 4), Tanh(), Linear(4, 1)])
    x = Tensor(np.random.randn(5, 2))
    out = net.forward(x)
    assert out.shape == (5, 1)


def test_neuralnet_params_and_zero_grad():
    net = NeuralNet([Linear(2, 3)])
    params = list(net.params())
    assert len(params) == 2
    x = Tensor(np.random.randn(1, 2), requires_grad=True)
    out = net.forward(x)
    out.sum().backward()
    net.zero_grad()
    for p in net.params():
        assert p.grad is None


def test_save_load_roundtrip_custom_extension(tmp_path):
    net = NeuralNet([Linear(2, 2)])
    path = str(tmp_path / "model.picodl")
    net.save(path)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".npz")

    net2 = NeuralNet([Linear(2, 2)])
    net2.load(path)
    for p1, p2 in zip(net.params(), net2.params()):
        assert np.allclose(p1.data, p2.data)


def test_save_load_architecture_mismatch(tmp_path):
    net = NeuralNet([Linear(2, 2)])
    path = str(tmp_path / "model.picodl")
    net.save(path)

    net_wrong = NeuralNet([Linear(3, 3)])
    try:
        net_wrong.load(path)
        assert False, "should have raised"
    except Exception:
        pass


def test_batch_iterator_shapes():
    inputs = np.random.randn(10, 2)
    targets = np.random.randn(10, 1)
    iterator = BatchIterator(batch_size=4, shuffle=False)
    batches = list(iterator(inputs, targets))
    assert len(batches) == 3
    assert batches[0].inputs.shape == (4, 2)
    assert batches[-1].inputs.shape == (2, 2)


def test_train_loop_regression_float_targets(capsys):
    net = NeuralNet([Linear(2, 4), Tanh(), Linear(4, 1)])
    inputs = np.random.randn(20, 2)
    targets = np.random.randn(20, 1)
    train(net, inputs, targets, num_epochs=2,
          iterator=BatchIterator(batch_size=5, shuffle=False),
          loss=MSE(), optimizer=SGD(lr=0.01))
    captured = capsys.readouterr()
    assert "0" in captured.out


def test_train_loop_classification_int_targets():
    net = NeuralNet([Linear(4, 8), GELU(), Linear(8, 3)])
    inputs = np.random.randn(20, 4)
    targets = np.random.randint(0, 3, size=20)
    train(net, inputs, targets, num_epochs=2,
          iterator=BatchIterator(batch_size=5, shuffle=False),
          loss=CrossEntropyLoss(), optimizer=AdamW(lr=0.01))


def test_train_accepts_tensor_input():
    net = NeuralNet([Linear(2, 4), Tanh(), Linear(4, 1)])
    inputs = Tensor(np.random.randn(10, 2))
    targets = Tensor(np.random.randn(10, 1))
    train(net, inputs, targets, num_epochs=1)
