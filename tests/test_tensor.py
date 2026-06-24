from jtorch.tensor import Tensor

def test_basic_tensor():
    x = Tensor([1, 2, 3])
    print("x:", x)

    y = Tensor([4, 5, 6])
    print("y:", y)

    z = x + y
    print("x + y =", z)

    m = x @ Tensor([[1], [1], [1]])
    print("x @ ones =", m)
