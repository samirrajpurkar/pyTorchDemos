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

# reshaping Tensors
# use the .view() method to reshape a tensor

x = torch.randn(2, 3, 4)
print('x', x)
print('reshape to 2 rows, 12 columns', x.view(2, 12))
print('same as above', x.view(2, -1))

# Computation Graphs
# and Automatic Differentiation

# Variables wrap tensor objects
x = autograd.Variable(torch.Tensor([1., 2., 3.]), requires_grad=True)
# you can access the data with the .data attribute
print('data in x Variable', x.data)

y = autograd.Variable(torch.Tensor([4., 5., 6.]), requires_grad=True)
print('data in y Variable', y.data)

z = x + y

print ('z = x+y', z.data)

print('But z knows something extra', z.creator)

s = z.sum()
print('s = z.sum()', s)
print('s.creator', s.creator)

print('calling backward() on any variable will run backprop, starting from it', s.backward())
print('ds / dx', x.grad)

# Understanding what is going on
# in the block below is crucial for being
# a successful programmer in deep learning

x = torch.randn((2, 2))
y = torch.randn((2, 2))
z = x + y

print('x', x)
print('y', y)
print('z', z)

var_x = autograd.Variable(x)
var_y = autograd.Variable(y)
var_z = var_x + var_y

print('var_z.creator', var_z.creator)

var_z_data = var_z.data
new_var_z = autograd.Variable(var_z_data)
print('new_var_z', new_var_z.creator)


print('-----------------------------------')
print('---Deep Learning Building Blocks---')
lin = nn.Linear(5, 3)
# maps from r^5 to r^3
# parameters A, b
# data is 2 x 5
# A maps from 5 to 3
print('lin.weight.data', lin.weight.data)
print('lin.bias.data', lin.bias.data)
x_v = autograd.Variable(torch.randn(2, 5))
print('x_v.data', x_v.data)
print('lin(data)', lin(x_v))

# Note that non-linearites typically
# don't have paramaters like affine maps do.
# That is, they don't have weights that
# are updated during training.

print('relu(x_v)', F.relu(x_v))

# Softmax is also in torch.functional
data_v = autograd.Variable(torch.randn(5))
print('data_v', data_v)
print('F.softmax(data_v)', F.softmax(data_v))
print('F.softmax(data_v).sum() = 1', F.softmax(data_v).sum())
print('There is also log_softmax(data)', F.log_softmax(data_v))


