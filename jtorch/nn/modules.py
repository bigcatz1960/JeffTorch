"""
modules.py
JeffTorch — Lightweight PyTorch‑shaped neural network library

This module defines the core building block of JeffTorch: the Module
base class. All neural network layers, containers, and trainable
components inherit from Module, mirroring the design of torch.nn.Module.

The goals of this implementation are:
    • clarity over complexity
    • explicit parameter tracking
    • predictable forward execution
    • minimal magic, maximum transparency

Module provides:
    • a standard forward() interface
    • automatic registration of parameters and submodules
    • a parameters() iterator for optimizers
    • a state_dict() skeleton for future serialization
    • a clean foundation for building custom layers

This file is intentionally small and readable so the mechanics of
parameter management and module composition are easy to understand,
modify, or extend. It forms the backbone of the JeffTorch learning
framework — a simple environment for exploring neural network design
without the abstraction layers of a full deep‑learning stack.
"""
