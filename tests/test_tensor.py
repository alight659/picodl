import numpy as np
from picodl.tensor import Tensor


def test_add_backward():
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)
    z = (x + y).sum()
    z.backward()
    assert np.allclose(x.grad, [1, 1])
    assert np.allclose(y.grad, [1, 1])


def test_mul_backward():
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0], requires_grad=True)
    z = (x * y).sum()
    z.backward()
    assert np.allclose(x.grad, [4, 5])
    assert np.allclose(y.grad, [2, 3])


def test_sub_neg():
    x = Tensor([5.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = (x - y).sum()
    z.backward()
    assert np.allclose(x.grad, [1])
    assert np.allclose(y.grad, [-1])


def test_pow():
    x = Tensor([3.0], requires_grad=True)
    z = (x ** 2).sum()
    z.backward()
    assert np.allclose(x.grad, [6])


def test_div():
    x = Tensor([6.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = (x / y).sum()
    assert np.allclose(z.data, [3])


def test_matmul_backward():
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    w = Tensor(np.random.randn(3, 4), requires_grad=True)
    out = x.matmul(w)
    out.sum().backward()
    assert x.grad.shape == (2, 3)
    assert w.grad.shape == (3, 4)


def test_sum_axis():
    x = Tensor(np.ones((2, 3)), requires_grad=True)
    out = x.sum(axis=1)
    out.sum().backward()
    assert np.allclose(x.grad, np.ones((2, 3)))


def test_mean():
    x = Tensor([2.0, 4.0, 6.0], requires_grad=True)
    m = x.mean()
    assert np.isclose(m.data, 4.0)
    m.backward()
    assert np.allclose(x.grad, [1 / 3, 1 / 3, 1 / 3])


def test_max():
    x = Tensor([1.0, 5.0, 3.0], requires_grad=True)
    m = x.max()
    assert np.isclose(m.data, 5.0)
    m.backward()
    assert np.allclose(x.grad, [0, 1, 0])


def test_argmax_no_grad():
    x = Tensor([[0.1, 0.7, 0.2]])
    assert x.argmax(axis=1)[0] == 1


def test_tanh_relu():
    x = Tensor([0.0, 1.0, -1.0], requires_grad=True)
    t = x.tanh()
    t.sum().backward()
    assert t.data.shape == (3,)

    x2 = Tensor([-1.0, 2.0], requires_grad=True)
    r = x2.relu()
    r.sum().backward()
    assert np.allclose(r.data, [0, 2])
    assert np.allclose(x2.grad, [0, 1])


def test_softmax_sums_to_one():
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    s = x.softmax(axis=-1)
    assert np.allclose(s.data.sum(axis=-1), 1.0)
    s.sum().backward()
    assert x.grad.shape == (4, 5)


def test_gelu():
    x = Tensor([0.0], requires_grad=True)
    g = x.gelu()
    assert np.isclose(g.data[0], 0.0)
    g.backward()


def test_log_clip():
    x = Tensor([1.0, np.e], requires_grad=True)
    l = x.log()
    assert np.allclose(l.data, [0, 1])
    c = x.clip(0.5, 2.0)
    assert np.allclose(c.data, [1.0, 2.0])


def test_sqrt():
    x = Tensor([4.0], requires_grad=True)
    s = x.sqrt()
    assert np.allclose(s.data, [2.0])
    s.backward()
    assert np.allclose(x.grad, [0.25])


def test_log_softmax():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    ls = x.log_softmax(axis=-1)
    assert np.allclose(np.exp(ls.data).sum(axis=-1), 1.0)
    ls.sum().backward()
    assert x.grad.shape == (3, 4)


def test_getitem_embedding_style():
    w = Tensor(np.arange(9.0).reshape(3, 3), requires_grad=True)
    rows = w[[0, 2]]
    assert rows.data.shape == (2, 3)
    rows.sum().backward()
    assert w.grad[1].sum() == 0
    assert w.grad[0].sum() > 0


def test_reshape_flatten():
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    y = x.reshape(2, 12)
    y.sum().backward()
    assert x.grad.shape == (2, 3, 4)
    f = x.flatten(start_dim=1)
    assert f.shape == (2, 12)


def test_transpose():
    x = Tensor(np.random.randn(2, 3), requires_grad=True)
    y = x.transpose(1, 0)
    assert y.shape == (3, 2)
    y.sum().backward()
    assert x.grad.shape == (2, 3)


def test_cat_and_split():
    a = Tensor(np.random.randn(2, 3), requires_grad=True)
    b = Tensor(np.random.randn(2, 3), requires_grad=True)
    c = Tensor.cat([a, b], axis=1)
    assert c.shape == (2, 6)
    c.sum().backward()
    assert a.grad.shape == (2, 3)
    assert b.grad.shape == (2, 3)

    parts = c.split(3, axis=1)
    assert len(parts) == 2
    assert parts[0].shape == (2, 3)


def test_stack():
    a = Tensor(np.random.randn(3), requires_grad=True)
    b = Tensor(np.random.randn(3), requires_grad=True)
    s = Tensor.stack([a, b], axis=0)
    assert s.shape == (2, 3)
    s.sum().backward()
    assert a.grad.shape == (3,)


def test_topk():
    x = Tensor([[1.0, 5.0, 3.0, 2.0]], requires_grad=True)
    vals, idx = x.topk(2, axis=1)
    assert np.allclose(vals.data, [[5.0, 3.0]])
    assert list(idx[0]) == [1, 2]
    vals.sum().backward()
    assert x.grad[0, 1] == 1
    assert x.grad[0, 2] == 1
    assert x.grad[0, 0] == 0


def test_scatter_index_add():
    base = Tensor(np.zeros((3, 4)), requires_grad=True)
    idx = np.array([[0], [1], [2]])
    src = Tensor(np.ones((3, 1)), requires_grad=True)
    out = base.scatter(1, idx, src)
    assert out.data[0, 0] == 1
    assert out.data[1, 1] == 1

    base2 = Tensor(np.zeros((3,)), requires_grad=True)
    src2 = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    idx2 = np.array([0, 0])
    added = base2.index_add(0, idx2, src2)
    assert added.data[0] == 3.0


def test_conv2d_shapes_and_padding_modes():
    x = Tensor(np.random.randn(1, 3, 8, 8), requires_grad=True)
    kernel = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
    out = x.conv2d(kernel, stride=1, padding=1)
    assert out.shape == (1, 4, 8, 8)
    out.sum().backward()
    assert x.grad.shape == (1, 3, 8, 8)
    assert kernel.grad.shape == (4, 3, 3, 3)

    for mode in ("reflect", "replicate", "circular"):
        out2 = x.conv2d(kernel, padding=1, pad_mode=mode)
        assert out2.shape == (1, 4, 8, 8)


def test_maxpool_avgpool():
    x = Tensor(np.random.randn(1, 2, 4, 4), requires_grad=True)
    mp = x.maxpool2d(2)
    assert mp.shape == (1, 2, 2, 2)
    mp.sum().backward()
    assert x.grad.shape == (1, 2, 4, 4)

    x2 = Tensor(np.ones((1, 2, 4, 4)), requires_grad=True)
    ap = x2.avgpool2d(2)
    assert np.allclose(ap.data, 1.0)


def test_multinomial_no_grad():
    probs = Tensor([[0.0, 1.0, 0.0]])
    samples = probs.multinomial(num_samples=1)
    assert samples[0, 0] == 1
