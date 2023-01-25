import torch 
from torch.autograd import Variable

# Define var to build computational graph
x = Variable(torch.Tensor([1.0, 2.0]).cuda(), requires_grad = True)
y = Variable(torch.Tensor([2.0, 3.0]).cuda(), requires_grad = True)
z = Variable(torch.Tensor([4.0, 3.0]).cuda(), requires_grad = True)

#Forward pass
a = x * y
b = a + z
c = torch.sum(b)

#compute grad
c.backward()

print(x.grad.data)
print(y.grad.data)
print(z.grad.data)