"""
activations.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module implements the core activation functions used throughout
JeffTorch. These are intentionally simple, NumPy‑based implementations
designed for clarity and educational value rather than performance.

The goal is to mirror the public API and behavior of torch.nn.functional
where it makes sense, while keeping the internals transparent so the
mechanics of each activation are easy to inspect, modify, or extend.

Included activations:
    • ReLU
    • Sigmoid
    • Tanh
    • (additional activations may be added as needed)

All functions operate on jtorch.Tensor objects and return new tensors
without modifying inputs in place.

This file is part of the JeffTorch learning framework — a minimal,
readable environment for experimenting with neural network components
and understanding how they work under the hood.
"""

from jtorch.nn.module import Module
import jtorch.nn.functional as F

class ReLU(Module):
    def forward(self, x):
        return F.relu(x)

class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)

class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)
