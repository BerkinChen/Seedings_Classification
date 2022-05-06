from dataset import TrainDataset, TestDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import SIFT, hog
from sklearn.svm import SVC
import joblib
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
        features = np.array(features)
        labels = np.array(labels)
    if feature == 'sift':
        """sift = SIFT()
        max_len = 0
        for X, y in data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            sift.detect_and_extract(X)
            x = sift.descriptors.reshape(-1).astype(np.float64)
            if x.shape[0] > max_len:
                max_len = x.shape[0]
            features.append(x)
            labels.append(y)
        features = np.array(features)
        features_ = np.array(np.resize(features[0], (max_len))).reshape(1,-1)
        for i in range(len(features)):
            features[i] = np.resize(features[i],(1,max_len))
            if i != 0:
                features_ = np.append(features_,features[i],axis=0)
        for X in valid_data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            sift.detect_and_extract(X)
            x = sift.descriptors.reshape(-1).astype(np.float64)
            valid_features.append(x)
        valid_features = np.array(valid_features)
        valid_features_ = np.array(
            np.resize(valid_features[0], (max_len))).reshape(1, -1)
        for i in range(len(valid_features)):
            valid_features[i] = np.resize(valid_features[i], (1, max_len))
            if i != 0:
               valid_features_ = np.append(
                   valid_features_, valid_features[i], axis=0)
        labels = np.array(labels)
        features = features_  
        valid_features = valid_features_"""
        sift = cv2.SIFT.create(100)
        for X, y in data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            _, des = sift.detectAndCompute(X, None)
            features.append(np.resize(des.reshape(-1), 12800))
            labels.append(y)
        for X in valid_data:
            X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
            _, des = sift.detectAndCompute(X, None)
            valid_features.append(np.resize(des.reshape(-1),12800))
        features = np.array(features)
        labels = np.array(labels)
    
    if classification == 'svm':
        clf = SVC(kernel='linear')
    
    
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


def train(model, loss_fn, optimizer,batch_szie=32, num_epochs=10,device='cpu', verbose=True):
    data = TrainDataset()
    train_dataset, test_dataset = train_test_split(data, test_size=0.2)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_szie,shuffle=True,num_workers=4)
    test_dataloader = DataLoader(test_dataset,batch_size=batch_szie,shuffle=False,num_workers=4)
    size = len(train_dataloader.dataset)
    for epoch in range(num_epochs):
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
                if i % 100 == 0:
                    loss, current = loss.item(), i * len(X)
                    print(f"loss: {loss:>7f}    [{current:>5d}/{size:>5d}]")
        if verbose is True:
            test(test_dataloader,model,loss_fn,device,verbose)


def test(dataloader, model, loss_fn, device='cpu', verbose=True):
    size = len(dataloader.dataset)
    batch_num = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            model.reconstruct()
            correct += (pred.argmax(1) == y).type(torch.float32).sum().item()

        test_loss /= batch_num
        correct /= size
        if verbose is True:
            print(
                f"Test Error: Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
