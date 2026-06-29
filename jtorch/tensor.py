# tensor.py
# JeffTorch — Core Tensor class
#
# This class wraps raw numerical data (NumPy arrays) and provides the
# foundation for all mathematical operations in JeffTorch.
#
# The initial implementation is intentionally minimal:
#   • stores data as a NumPy array
#   • provides shape/dtype helpers
#   • stubs out autograd fields (grad, requires_grad)
#   • stubs out operator overloads for future math support
#
# As JeffTorch evolves, this class will grow to include:
#   • arithmetic operations (add, matmul, etc.)
#   • broadcasting rules
#   • autograd graph construction
#   • backward() implementation
#   • device/dtype management
#
# For now, Tensor is a lightweight container that mirrors the public
# behavior of torch.Tensor where practical, without implementing the
# full PyTorch backend.

import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        # Convert input to NumPy array
        self.data = np.array(data, dtype=float)

        # Autograd fields (future use)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = None     # function to compute gradients
        self._prev = set()        # previous Tensors in the graph

    # ------------------------------
    # Basic properties
    # ------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def T(self):
        return Tensor(self.data.T)

    # ------------------------------
    # Representation
    # ------------------------------
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"

    # ------------------------------
    # Math Operators
    # ------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        out._prev = {self, other}

        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad @ other.data.T
            if other.requires_grad:
                other.grad = other.grad + self.data.T @ out.grad

        out._backward = _backward
        return out
    
    # ------------------------------
    # Autograd engine
    # ------------------------------

    def backward(self):
        # seed gradient
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # topological sort
        topo = []
        visited = set()

        def build_topo(v):
            """
            Build a topological ordering of all Tensors that lead to `v`.

            This performs a DFS over the autograd graph:
            • visits each node exactly once
            • ensures parents appear before children
            • produces a list used for backward traversal
            """
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # go backwards
        for node in reversed(topo):
            if node._backward:
                node._backward()
