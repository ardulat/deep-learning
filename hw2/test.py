# imports
from pylab import *
from ConvNet import Net

import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

net = Net()
net.load_state_dict(torch.load('ConvNet.pt'))

if torch.cuda.is_available():
    net.cuda()

def read_images(dir_path):
    
    X = []
    y = []
    
    # define a label map
    labelmap = {
                'airplane': 0,
                'bird': 1,
                'dog': 2,
                'frog': 3,
                'horse': 4,
                'apple': 5,
                'grape': 6,
                'kiwi': 7,
                'lemon': 8,
                'strawberry': 9
               }
    
    directory_list = os.listdir(dir_path)
    # remove OS X's .DS_Store file
    if '.DS_Store' in directory_list:
        directory_list.remove('.DS_Store')
    
    for i, class_name in enumerate(directory_list):
        for j, image_name in enumerate(os.listdir(dir_path+class_name)):
            image_path = dir_path+class_name+'/'+image_name
            image = cv2.imread(image_path)
            X.append(image)
            y.append(labelmap[class_name])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y




def load_data(dir_path1, dir_path2):
    
    X1, y1 = read_images(dir_path1)
    X2, y2 = read_images(dir_path2)
    
    X2_resized = np.zeros((X2.shape[0], 32, 32, X2.shape[3]), dtype=np.uint8)
    
    for i in range(X2.shape[0]):
        X2_resized[i,:,:,0] = cv2.resize(X2[i,:,:,0], (32,32))
        X2_resized[i,:,:,1] = cv2.resize(X2[i,:,:,1], (32,32))
        X2_resized[i,:,:,2] = cv2.resize(X2[i,:,:,2], (32,32))
    
    X = np.append(X1, X2_resized, axis=0)
    y = np.append(y1, y2, axis=0)
    
    return X, y



class CIFAR10(torch.utils.data.dataset.Dataset):
    __Xs = None
    __ys = None
    
    def __init__(self, dir_path1, dir_path2, transform=None):
        self.transform = transform
        self.__Xs, self.__ys = load_data(dir_path1, dir_path2)
        
    def __getitem__(self, index):
        img = self.__Xs[index]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert image and label to torch tensors
        img = torch.from_numpy(np.asarray(img))
        label = torch.from_numpy(np.asarray(self.__ys[index]))
        
        return img, label
    
    def __len__(self):
        return self.__Xs.shape[0]


batch_size = 128

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(device)
torch.cuda.empty_cache()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


# In[ ]:


test_dir_path1 = 'hw2 data/data1/test/'
test_dir_path2 = 'hw2 data/data2/test/'
batch_size = 1

testset = CIFAR10(test_dir_path1, test_dir_path2, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=1)


# In[ ]:


correct = 0
total = 0

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








