from jtorch.tensor import Tensor
from jtorch.nn.linear import Linear

def test_linear_forward():
    layer = Linear(3, 2)
    x = Tensor([1.0, 2.0, 3.0])
    y = layer(x)
    print("Output:", y)

if __name__ == "__main__":
    test_linear_forward()
