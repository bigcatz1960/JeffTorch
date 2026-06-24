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

    # ------------------------------
    # Representation
    # ------------------------------
    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"

    # ------------------------------
    # Operator stubs (future math)
    # ------------------------------
    def __add__(self, other):
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            # allow adding scalars
            other_data = other

        # For now, enforce identical shapes (no broadcasting yet)
        if isinstance(other_data, np.ndarray) and other_data.shape != self.data.shape:
            raise ValueError(f"Shape mismatch in add: {self.data.shape} vs {other_data.shape}")

        return Tensor(self.data + other_data)

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError("Matmul requires another Tensor")

        return Tensor(self.data @ other.data)
    
    @property
    def T(self):
        return Tensor(self.data.T)

    def backward(self):
        raise NotImplementedError("Autograd not implemented yet")
