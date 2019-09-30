import torch
bl = []
a = torch.Tensor([1,2])
a.requires_grad=True
b = a*a
b.register_hook(print)
bl.append(b)
b = b.sum()
b.backward()
print(bl[0])