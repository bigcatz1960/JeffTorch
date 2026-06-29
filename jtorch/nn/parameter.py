
# parameter.py
# JeffTorch: Minimal neural network framework inspired by PyTorch
#
# The Parameter class is a lightweight wrapper around a Tensor.
# It marks a value as trainable and provides a place to store gradients.
#
# In JeffTorch, Parameters are detected automatically by Module.__setattr__,
# which registers them inside the module's internal _parameters dictionary.
#
# This mirrors the behavior of torch.nn.Parameter, but remains intentionally
# simple: a Parameter contains only two fields:
#   - value : the underlying Tensor (or numeric data)
#   - grad  : the gradient accumulated during backpropagation
#
# Parameters do not implement math operations. All computation happens on
# the underlying Tensor. Parameters exist solely to represent trainable
# state within Modules.

from jtorch.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, value):
        super().__init__(value, requires_grad=True)
        self.grad = None

        # Gradient placeholder
        self.grad = None

    def __repr__(self):
        return f"Parameter(data={self.data}, grad={self.grad})"
