from pylab import *
from SimpleNet import Net

import torch

net = torch.load('SimpleNet.pt')
net.eval()