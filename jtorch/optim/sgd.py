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
for p in model.parameters():
    p.data -= lr * p.grad
