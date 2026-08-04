
# picodl.

A tiny deep learning library built from scratch in Python - autograd engine, layers, losses, optimizers, and a training loop, all on top of plain numpy. No PyTorch, no TensorFlow, no hidden framework underneath.


## Documentation

[Documentation](https://picodl.vercel.app/docs.html)


## Installation

Install picodl with pip

```bash
  pip install picodl
```
    
## Running Tests

To run tests, run the following command

```bash
    # requires pytest, pip install pytest
    pytest -v
```


## Demo

A sample model

```python
from picodl.nn import NeuralNet
from picodl.layers import Linear, GELU
from picodl.loss import CrossEntropyLoss
from picodl.optim import AdamW
from picodl.train import train

net = NeuralNet([
    Linear(784, 128),
    GELU(),
    Linear(128, 10),
])

train(net, x_train, y_train,
      num_epochs=10,
      loss=CrossEntropyLoss(),
      optimizer=AdamW(lr=0.001, weight_decay=0.01))

net.save("model.picodl")
```
## Authors

- [@alight659](https://www.github.com/alight659)


## License

[MIT](./LICENSE)

