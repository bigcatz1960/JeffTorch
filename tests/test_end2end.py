from jtorch.tensor import Tensor
from jtorch.nn.linear import Linear
from jtorch.nn.activations import ReLU
from jtorch.nn.container import Sequential
from jtorch.optim.sgd import SGD

# 1. Build a tiny model
model = Sequential(
    Linear(4, 8),
    ReLU(),
    Linear(8, 1)
)

# 2. Create an input
x = Tensor.randn(4)

# 3. Forward pass
y = model(x)
print("Output:", y.data)

# 4. Backward pass
y.backward()

# 5. Check gradients
print("\nGradients:")
for p in model.parameters():
    print(p.grad)

# 6. Optimizer step
opt = SGD(model.parameters(), lr=0.01)
opt.step()
opt.zero_grad()

print("\nParameter update complete.")
