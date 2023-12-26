import torch

x = torch.randn(5, requires_grad=True)
print(x)
print(x.requires_grad)

print()

val = x[0]

print(val)
print(val.requires_grad)
