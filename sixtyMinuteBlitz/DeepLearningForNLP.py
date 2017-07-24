# Let us look what we can do with tensors
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# Create a torch Tensor object with the
# given data.
# It is a 1D vector

V_data = [1., 2., 3.]
V = torch.Tensor(V_data)

print('V Tensor', V)

# Create a matrix
M_data = [
    [1., 2., 3.],
    [4., 5., 6]
]

M = torch.Tensor(M_data)
print('Matrix M', M)

# Create a 3D tensor of size 2x2x2
T_data = [
    [[1., 2.], [3., 4.]],
    [[5., 6.], [7., 8.]]
]
T = torch.Tensor(T_data)
print('3D Tensor', T)

# Indexing
print('Index into a V & get a scalar', V[0])
print('Index into a M & get a vector', M[0])
print('Index into a 3D Tensor & get a matrix', T[0])

# You can create a tensor with random data
# and the supplied dimensionality

r3d = torch.randn(3, 4, 5)

# Operations with Tensors
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])
z = x + y
print(z)

# One helpful operation that we will make
# use of later is concatenation

# By default, it concatenates along
# the first axis (concatenates rows)
x_1 = torch.randn(2, 5)
print('x_1', x_1)
y_1 = torch.randn(3, 5)
print('y_1', y_1)
z_1 = torch.cat([x_1, y_1])
print(z_1)

# Concatenates columns:
x_2 = torch.randn(2, 3)
print('x_2', x_2)
y_2 = torch.randn(2, 5)
print('y_2', y_2)
z_2 = torch.cat([x_2, y_2], 1)
print('z_2', z_2)


