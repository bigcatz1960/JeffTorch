"""
linear.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module implements the Linear (fully connected) layer, one of the
fundamental building blocks of neural networks. It mirrors the behavior
and public API of torch.nn.Linear while remaining intentionally simple
and transparent for learning and experimentation.

A Linear layer performs an affine transformation:

        y = x @ W.T + b

where:
    • W is a learnable weight matrix of shape (out_features, in_features)
    • b is an optional learnable bias vector of shape (out_features,)
    • x is the input tensor

The goals of this implementation are:
    • clear, readable math
    • explicit parameter initialization
    • predictable forward behavior
    • seamless integration with Module, optimizers, and functional ops

This file is part of the JeffTorch learning framework — a minimal,
NumPy‑backed environment for understanding how neural network layers
work internally and how higher‑level architectures are constructed from
simple, composable components.
"""
import numpy as np
from jtorch.tensor import Tensor
from jtorch.nn.parameter import Parameter
from jtorch.nn.module import Module

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        # Xavier-like initialization (simple version)
        limit = 1.0 / np.sqrt(in_features)

        self.weight = Parameter(
            np.random.uniform(-limit, limit, (out_features, in_features))
        )

        self.bias = Parameter(
            np.zeros(out_features)
        )

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x @ self.weight.T + self.bias
