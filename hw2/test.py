from pylab import *
from ConvNet import Net

import torch

net = Net()
net.load_state_dict(torch.load('ConvNet.pt'))

if torch.cuda.is_available():
    net.cuda()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the %d test images: %d %%' % (total,
    100 * correct / total))