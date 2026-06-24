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
from jtorch.nn.parameter import Parameter

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        # Intercept assignments to register parameters and submodules
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        # Always set the attribute normally
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        # Wrap forward() so you can call the module like a function
        return self.forward(*args, **kwargs)

    def parameters(self):
        # Yield this module's parameters
        for p in self._parameters.values():
            yield p

        # Recursively yield child module parameters
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self):
        # Zero this module's parameters
        for p in self._parameters.values():
            p.grad = None

        # Recursively zero child module parameters
        for m in self._modules.values():
            m.zero_grad()


