import numpy as np
from picodl.tensor import Tensor
from picodl.nn import NeuralNet
from picodl.loss import Loss, MSE
from picodl.optim import Optimizer, SGD
from picodl.data import DataIterator, BatchIterator


def train(net: NeuralNet, inputs: Tensor, targets: Tensor, num_epochs: int = 1000,
          iterator: DataIterator = BatchIterator(), loss: Loss = MSE(), optimizer: Optimizer = SGD()) -> None:
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