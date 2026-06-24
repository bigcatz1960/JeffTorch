"""
functional.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module implements the stateless, function‑based neural network
operations used throughout JeffTorch. It mirrors the design philosophy
of torch.nn.functional: pure mathematical transforms with no internal
state, no parameters, and no side effects.

Functional operations are the low‑level building blocks behind many
layers and loss functions. They provide a clear, explicit view of the
math that drives neural networks, making them ideal for learning,
debugging, and experimentation.

Included operations:
    • mse_loss
    • relu
    • sigmoid
    • tanh
    • (additional ops may be added as needed)

All functions operate on jtorch.Tensor objects and return new tensors.
Nothing is modified in place unless explicitly documented.

This file is part of the JeffTorch learning framework — a minimal,
transparent environment for exploring neural network mechanics and
understanding how higher‑level modules are built from simple, composable
functional primitives.
"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)
