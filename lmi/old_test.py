import torch
import lmi

print(torch.__version__)

print(lmi.__file__)

t = torch.tensor([[1., -1.], [1., -1.]])
print(t)
print(lmi.add_one(t))

print(lmi.tch_test())