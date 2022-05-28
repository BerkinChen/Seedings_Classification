from torchvision.models import resnet18,vgg11_bn
from torch import nn

class Resnet(nn.Module):
    def __init__(self,device='cpu',dropout=False):
        super(Resnet, self).__init__()
        self.net = resnet18()
        self.net.conv1.kernel_size = (3,3)
        self.net.add_module('relu',nn.ReLU(inplace=True))
        if dropout:
            self.net.add_module('dropout',nn.Dropout())
        self.net.add_module('fc1',nn.Linear(1000,12))
        self.net.to(device)
        
    def forward(self,X):
        return self.net(X)
   
class Vgg(nn.Module):
    def __init__(self, device='cpu',dropout=False):
        super(Vgg, self).__init__()
        self.net = vgg11_bn()
        if dropout:
            self.net.classifier[6].out_features = 12
        else:
            self.net.classifier = nn.Sequential(
                nn.Linear(25088,4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096,12))
        self.net.to(device)
        
    def forward(self,X):
        return self.net(X) 
    
class MLP(nn.Module):
    def __init__(self, device='cpu',dropout=False):
        super(MLP, self).__init__()
        if dropout:
            self.net = nn.Sequential(
                nn.Conv2d(3,64,3,1,1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(32*14*14,1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024,512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512,12))
        else:
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(32*14*14, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 12))
        self.net.to(device)
    
    def forward(self,x):
        x = self.net(x)
        return x
