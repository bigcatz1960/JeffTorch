"""
tensor.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module defines the core Tensor class used throughout JeffTorch.
It provides a minimal, NumPy‑backed abstraction for numerical data,
mirroring the public behavior of torch.Tensor where practical while
remaining small, transparent, and easy to reason about.

The goals of this implementation are:
    • clarity over performance
    • explicit, readable mechanics
    • a foundation for experimentation
    • compatibility with the JeffTorch Module and functional layers

Tensor supports:
    • basic arithmetic operations
    • shape and dtype inspection
    • simple broadcasting behavior
    • gradient placeholders (for future autograd work)
    • conversion to and from NumPy arrays

This file is intentionally lightweight. It exposes the essential
building blocks needed to understand how tensor libraries work under
the hood, without the complexity of CUDA kernels, dispatch layers,
or a full autograd engine.

As JeffTorch evolves, this class serves as the central point where new
ideas about tensor semantics, gradient behavior, or experimental
features can be explored and prototyped.
"""
