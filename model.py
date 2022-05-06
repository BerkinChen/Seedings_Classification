from torchvision.models import resnet18,vgg11
from torch import nn

class Resnet(nn.Module):
    def __init__(self,device):
        super(Resnet, self).__init__()
        self.net = resnet18()
        self.net.fc.out_features = 12
        self.net.to(device)
        
    def forward(self,X):
        return self.net(X)
   
class Vgg(nn.Module):
    def __init__(self, device):
        super(Vgg, self).__init__()
        self.net = vgg11()
        self.net.classifier[6].out_features = 12
        self.net.to(device)
        
    def forward(self,X):
        return self.net(X) 
