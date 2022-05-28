import warnings
from model import Resnet, Vgg, MLP
from train import train, get_results, traditional_method
from torch import nn, optim
import torch
from torchvision import transforms
from seed import setup_seed
import argparse
warnings.filterwarnings('ignore')

args = argparse.ArgumentParser()
args.add_argument('-m', '--mode', default='deep', dest='mode', type=str)
args.add_argument('-d', '--device', default='cpu', dest='device', type=str)
args.add_argument('-n', '--net', default='Resnet', dest='net', type=str)
args.add_argument('-dropout', default=False,
                  dest='dropout', action='store_true')
args.add_argument('-optim', default='Adam', dest='optim', type=str)
args.add_argument('-weightdecay', default=False,
                  dest='weightdecay', action='store_true')
args.add_argument('-t', '--transform', default=None,
                  dest='transform', type=str)
args = args.parse_args()
setup_seed(1)

if args.mode != 'deep':
    feature, classification = args.mode.split('+')
    traditional_method(feature, classification)

if args.mode == 'deep':
    device = args.device
    if args.dropout is False:
        if args.net == 'Resnet':
            net = Resnet(device)
        elif args.net == 'Vgg':
            net = Vgg(device)
        elif args.net == 'MLP':
            net = MLP(device)
    else:
        if args.net == 'Resnet':
            net = Resnet(device, dropout=True)
        elif args.net == 'Vgg':
            net = Vgg(device, dropout=True)
        elif args.net == 'MLP':
            net = MLP(device, dropout=True)

    loss = nn.CrossEntropyLoss()
    if args.weightdecay is False:
        if args.optim == 'Adam':
            optimizer = optim.Adam(net.parameters(), lr=1e-3)
        elif args.optim == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
        elif args.optim == 'Adagrad':
            optimizer = optim.Adagrad(net.parameters(), lr=1e-3)
    else:
        if args.optim == 'Adam':
            optimizer = optim.Adam(
                net.parameters(), lr=1e-3, weight_decay=1e-4)
        elif args.optim == 'SGD':
            optimizer = optim.SGD(net.parameters(), lr=1e-2,
                                  momentum=0.9, weight_decay=1e-4)
        elif args.optim == 'Adagrad':
            optimizer = optim.Adagrad(
                net.parameters(), lr=1e-3, weight_decay=1e-4)
    if args.transform is None:
        transform = transforms.Compose([transforms.ToTensor()])
    elif args.transform == 'normalize':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([
                                       0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif args.transform == 'randomflip':
        transform = transforms.Compose([transforms.ToTensor(
        ), transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()])
    elif args.transform == 'randomcrop':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.RandomResizedCrop((224, 224))])

    filename = 'checkpoint/checkpoint_'
    resultname = 'submission/'
    filename += (net._get_name()+'_')
    resultname += (net._get_name()+'_')
    filename += (args.optim+'_')
    resultname += (args.optim+'_')
    if args.dropout:
        filename += 'dropout_'
        resultname += 'dropout_'
    if args.weightdecay:
        filename += 'weightdecay_'
        resultname += 'weightdecay_'
    if args.transform is not None:
        filename += (args.transform + '_')
        resultname += (args.transform + '_')
    filename += '.pt'
    resultname += 'submission.csv'

    net.train()
    train(net, loss, optimizer, filename, device=device,
          transform=transform, verbose=True)
    net.eval()
    get_results(net,  filename, resultname, device, transform=transform)
