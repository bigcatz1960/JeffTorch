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
from jtorch.tensor import Tensor

# ------------------------------
# Activation functions
# ------------------------------

def relu(x: Tensor):
    """ReLU activation: max(0, x)"""
    out_data = np.maximum(0, x.data)
    out = Tensor(out_data, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            if x.grad is None:
                x.grad = 0.0
            x.grad += (x.data > 0) * out.grad

    out._backward = _backward
    return out

def sigmoid(x: Tensor):
    s = 1 / (1 + np.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            if x.grad is None:
                x.grad = 0.0
            x.grad += (s * (1 - s)) * out.grad

    out._backward = _backward
    return out

def tanh(x: Tensor):
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            if x.grad is None:
                x.grad = 0.0
            x.grad += (1 - t**2) * out.grad

    out._backward = _backward
    return out

# ------------------------------
# Loss functions
# ------------------------------
# Mean squared error loss: mean((pred - target)^2)
 
def mse_loss(pred: Tensor, target: Tensor):
    diff = pred.data - target.data
    out_data = np.mean(diff ** 2)

    out = Tensor(out_data, requires_grad=pred.requires_grad or target.requires_grad)
    out._prev = {pred, target}

    def _backward():
        N = pred.data.size

        if pred.requires_grad:
            if pred.grad is None:
                pred.grad = 0.0
            pred.grad += (2 * diff / N) * out.grad

        if target.requires_grad:
            if target.grad is None:
                target.grad = 0.0
            target.grad += (-2 * diff / N) * out.grad

    out._backward = _backward
    return out
