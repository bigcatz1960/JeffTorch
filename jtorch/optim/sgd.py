"""
sgd.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module implements a minimal Stochastic Gradient Descent (SGD)
optimizer. It mirrors the basic behavior of torch.optim.SGD while
remaining intentionally simple and transparent for learning and
experimentation.

SGD updates parameters according to:

        param = param - lr * grad

Optional features such as momentum, weight decay, or dampening may be
added later, but the core implementation focuses on clarity and the
mechanics of parameter updates.

The optimizer expects:
    • an iterable of parameters (from Module.parameters())
    • each parameter to expose a `.grad` field
    • gradients to be accumulated externally (e.g., during backprop)

This file is part of the JeffTorch learning framework — a minimal,
readable environment for understanding how optimizers interact with
modules, parameters, and the training loop at a fundamental level.
"""
# sgd.py
# JeffTorch — Minimal Stochastic Gradient Descent optimizer

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        # Apply gradient descent update
        for p in self.parameters:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

    def zero_grad(self):
        # Reset gradients to None
        for p in self.parameters:
            p.grad = None
