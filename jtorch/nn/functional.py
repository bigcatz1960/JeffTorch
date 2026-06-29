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
from tensor import Tensor   # adjust import path if needed


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
            # derivative: 1 where x > 0, else 0
            x.grad = x.grad + (x.data > 0) * out.grad

    out._backward = _backward
    return out


def sigmoid(x: Tensor):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    s = 1 / (1 + np.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            # derivative: s * (1 - s)
            x.grad = x.grad + (s * (1 - s)) * out.grad

    out._backward = _backward
    return out


def tanh(x: Tensor):
    """Tanh activation: (e^x - e^-x) / (e^x + e^-x)"""
    t = np.tanh(x.data)
    out = Tensor(t, requires_grad=x.requires_grad)
    out._prev = {x}

    def _backward():
        if x.requires_grad:
            # derivative: 1 - tanh(x)^2
            x.grad = x.grad + (1 - t**2) * out.grad

    out._backward = _backward
    return out


# ------------------------------
# Loss functions
# ------------------------------

def mse_loss(pred: Tensor, target: Tensor):
    """Mean squared error loss: mean((pred - target)^2)"""
    diff = pred.data - target.data
    out_data = np.mean(diff ** 2)

    out = Tensor(out_data, requires_grad=pred.requires_grad or target.requires_grad)
    out._prev = {pred, target}

    def _backward():
        # d/dpred = 2*(pred - target)/N
        N = pred.data.size
        if pred.requires_grad:
            pred.grad = pred.grad + (2 * diff / N) * out.grad
        if target.requires_grad:
            # d/dtarget = -2*(pred - target)/N
            target.grad = target.grad + (-2 * diff / N) * out.grad

    out._backward = _backward
    return out
