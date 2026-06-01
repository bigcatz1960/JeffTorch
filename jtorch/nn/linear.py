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
