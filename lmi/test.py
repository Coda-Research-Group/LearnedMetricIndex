import torch
import lmi

print(torch.__version__)

print(lmi.__file__)

l = lmi.Lmi(320, 768)
t = torch.randn(10000, 768)
l.train(t)
