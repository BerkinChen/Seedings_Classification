import os
from dataset import TrainDataset, TestDataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from skimage.feature import hog
from scipy.optimize import linear_sum_assignment
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

target_names = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',
                'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']


def traditional_method(feature='hog', classification='svm'):
    data = TrainDataset()
    valid_data = TestDataset()
    features = []
    valid_features = []
    labels = []
    if feature == 'hog':
        for X, y in data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            features.append(hog(X))
            labels.append(y)
        for X in valid_data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            valid_features.append(hog(X))
    if feature == 'sift':
        sift = cv2.SIFT.create(200)
        for X, y in data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            _, des = sift.detectAndCompute(X, None)
            features.append(np.resize(des, 25600))
            labels.append(y)
        for X in valid_data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            _, des = sift.detectAndCompute(X, None)
            valid_features.append(np.resize(des,25600))
    features = np.array(features)
    valid_features = np.array(valid_features)
    labels = np.array(labels)
    
    if classification == 'svm':
        clf = SVC(kernel='linear')
    if classification == 'kernel_svm':
        clf = SVC(kernel='sigmoid')
    if classification == 'kmeans':
        clf = KMeans(12)
    
    if classification == 'kmeans':
        train_pred = clf.fit_predict(features.astype('double'))
        cost = np.zeros((12, 12))
        for i in range(len(labels)):
            cost[train_pred[i]][labels[i]] += 1
        _, index = linear_sum_assignment(cost, maximize=True)
        pred = clf.predict(valid_features.astype('double'))
        new_pred = np.zeros(pred.shape)
        for i in range(len(pred)):
            new_pred[i] = index[pred[i]]
        pred = new_pred.astype(np.int32)
    else:
        clf.fit(features,labels)
        pred = clf.predict(valid_features)
    result = []
    for i in range(len(pred)):
        result.append(target_names[pred[i]])
    valid_data.df['species'] = result
    valid_data.df['file'] = valid_data.df['image_id']
    valid_data.df[['file', 'species']].to_csv(
        'submission/'+feature+'_'
        + classification+'_'+'submission.csv', index=False)


def train(model, loss_fn, optimizer,filename,batch_szie=64, num_epochs=50,device='cpu', verbose=True,transform=None):
    data = TrainDataset(transform=transform)
    train_dataset, test_dataset = train_test_split(data, test_size=0.2,random_state=0)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_szie,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_szie,shuffle=False,num_workers=4)
    size = len(train_dataloader.dataset)
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        if verbose is True:
            print(f"Epoch {epoch+1}\n-------------------------------")
        for i, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose is True:
                if i % 10 == 0:
                    loss, current = loss.item(), i * len(X)
                    print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")
        acc = test(test_dataloader,model,loss_fn,device,verbose = verbose)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), filename)


def test(dataloader, model, loss_fn, device='cpu', verbose=True):
    model.eval()
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()

        test_loss /= batch_num
        correct /= size
        if verbose is True:
            print(
                f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

def get_results(model,filename,resultname,device='cpu',transform=None):
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
    valid_data = TestDataset(transform=transform)
    valid_dataloader = DataLoader(dataset=valid_data,batch_size=32)
    result = []
    model.eval()
    for X in valid_dataloader:
        X = X.to(device)
        pred = model(X)
        
        pred = pred.argmax(1)
        for i in range(len(pred)):
            result.append(target_names[pred[i]])
    valid_data.df['species'] = result
    valid_data.df['file'] = valid_data.df['image_id']
    valid_data.df[['file', 'species']].to_csv(resultname, index=False)
        