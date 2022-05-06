from train import traditional_method
import warnings
from model import Resnet, Vgg
from train import train,get_results
from torch import nn,optim
import torch
from torchvision import transforms
from seed import setup_seed
warnings.filterwarnings('ignore')

setup_seed(1)
#traditional_method('hog','svm')
#traditional_method('sift','svm')

device = 'cuda:1'
net = Vgg(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 1e-3)
transform = transforms.Compose([transforms.ToTensor()])

train(net,loss,optimizer,batch_szie=128,device=device,transform=transform,verbose=True)
torch.save(net.state_dict(), 'checkpoint/checkpoint_'+net._get_name()+'.pt')
net.load_state_dict(torch.load('checkpoint/checkpoint_'+net._get_name()+'.pt'))
get_results(net, device, transform=transform)

