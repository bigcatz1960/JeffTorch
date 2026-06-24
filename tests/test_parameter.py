from jtorch.nn.parameter import Parameter

def test_parameter_inherits_tensor():
    p = Parameter([1, 2, 3])
    print("Parameter:", p)
    print("Is Tensor:", isinstance(p, Parameter))
