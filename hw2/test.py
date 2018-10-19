from pylab import *
from SimpleNet import Net

import torch

net = torch.load('VGG.pt')
net.eval()