import numpy as np
from picodl.tensor import Tensor
from picodl.layers import (
    Linear, Tanh, ReLU, GELU, Conv2D, Embedding, TiedLinear, LayerNorm,
    MaxPool2D, AvgPool2D, Flatten, BatchNorm2D, Dropout, GlobalAvgPool,
)


def test_linear_forward_shape():
    layer = Linear(4, 2)
    x = Tensor(np.random.randn(5, 4))
    out = layer.forward(x)
    assert out.shape == (5, 2)


def test_linear_backward_updates_grad():
    layer = Linear(3, 2)
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    out = layer.forward(x)
    out.sum().backward()
    assert layer.params["w"].grad is not None
    assert layer.params["b"].grad is not None


def test_tanh_relu_gelu_layers():
    x = Tensor([-1.0, 0.0, 1.0])
    assert Tanh().forward(x).shape == (3,)
    r = ReLU().forward(x)
    assert np.allclose(r.data, [0, 0, 1])
    g = GELU().forward(x)
    assert np.isclose(g.data[1], 0.0)


def test_conv2d_layer():
    conv = Conv2D(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(2, 3, 8, 8), requires_grad=True)
    out = conv.forward(x)
    assert out.shape == (2, 4, 8, 8)
    out.sum().backward()
    assert conv.params["w"].grad is not None
    assert conv.params["b"].grad is not None


def test_conv2d_layer_pad_mode():
    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, padding=1, pad_mode="reflect")
    x = Tensor(np.random.randn(1, 2, 6, 6), requires_grad=True)
    out = conv.forward(x)
    assert out.shape == (1, 3, 6, 6)


def test_embedding_layer():
    emb = Embedding(vocab_size=10, embed_dim=4)
    idx = [1, 3, 5]
    out = emb.forward(idx)
    assert out.shape == (3, 4)
    out.sum().backward()
    assert emb.params["w"].grad is not None
    assert emb.params["w"].grad[0].sum() == 0
    assert emb.params["w"].grad[1].sum() != 0


def test_tied_linear():
    emb = Embedding(vocab_size=10, embed_dim=4)
    tied = TiedLinear(emb)
    hidden = emb.forward([1, 2])
    logits = tied.forward(hidden)
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert emb.params["w"].grad is not None


def test_tied_linear_bias():
    emb = Embedding(vocab_size=6, embed_dim=4)
    tied = TiedLinear(emb, bias=True)
    assert "b" in tied.params
    out = tied.forward(emb.forward([0, 1]))
    assert out.shape == (2, 6)


def test_layernorm():
    ln = LayerNorm(dim=6)
    x = Tensor(np.random.randn(3, 6), requires_grad=True)
    out = ln.forward(x)
    assert out.shape == (3, 6)
    out.sum().backward()
    assert ln.params["gamma"].grad is not None
    assert ln.params["beta"].grad is not None


def test_maxpool_avgpool_layers():
    x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
    mp_out = MaxPool2D(2).forward(x)
    assert mp_out.shape == (1, 2, 2, 2)

    x2 = Tensor(np.ones((1, 2, 4, 4)))
    ap_out = AvgPool2D(2).forward(x2)
    assert np.allclose(ap_out.data, 1.0)


def test_flatten_layer():
    x = Tensor(np.random.randn(2, 3, 4, 4))
    out = Flatten().forward(x)
    assert out.shape == (2, 48)


def test_global_avg_pool():
    x = Tensor(np.ones((2, 3, 5, 5)), requires_grad=True)
    out = GlobalAvgPool().forward(x)
    assert out.shape == (2, 3)
    assert np.allclose(out.data, 1.0)
    out.sum().backward()
    assert np.allclose(x.grad, 1.0 / 25.0)


def test_batchnorm_train_eval():
    bn = BatchNorm2D(num_features=3)
    x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=True)
    out = bn.forward(x)
    assert out.shape == (4, 3, 5, 5)
    out.sum().backward()

    bn.eval()
    out_eval = bn.forward(x)
    assert out_eval.shape == (4, 3, 5, 5)
    bn.train()


def test_dropout_train_eval():
    do = Dropout(p=0.5)
    x = Tensor(np.ones((10, 10)))
    out_train = do.forward(x)
    assert out_train.shape == (10, 10)

    do.eval()
    out_eval = do.forward(x)
    assert np.allclose(out_eval.data, x.data)
    do.train()


def test_dropout_zero_p_is_identity():
    do = Dropout(p=0.0)
    x = Tensor(np.random.randn(3, 3))
    out = do.forward(x)
    assert np.allclose(out.data, x.data)
