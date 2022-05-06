import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
import torch
from torchvision import transforms


class Data():
    def __init__(self,is_train=True):
        classes = {'Black-grass': 0,
                   'Charlock': 1,
                   'Cleavers': 2,
                   'Common Chickweed': 3,
                   'Common wheat': 4,
                   'Fat Hen': 5,
                   'Loose Silky-bent': 6,
                   'Maize': 7,
                   'Scentless Mayweed': 8,
                   'Shepherds Purse': 9,
                   'Small-flowered Cranesbill': 10,
                   'Sugar beet': 11}
        if is_train:
            self.data = pd.DataFrame(columns=['image_path', 'label', 'image_id'])
            self.data.astype({'label': np.int32})
            pathToTrainData = './data/train'
            for dirname, _, filenames in os.walk(pathToTrainData):
                for filename in filenames:
                    path = os.path.join(dirname, filename)
                    class_label = dirname.split('/')[-1]
                    class_label = classes[class_label]
                    image_id = filename

                    self.data = self.data.append(
                        {'image_id': image_id, 'image_path': path, 'label': class_label}, ignore_index=True)
        else:
            self.data = pd.DataFrame(columns=['image_id','image_path'])
            pathToTestData = './data/test'
            for dirname, _, filenames in os.walk(pathToTestData):
                for filename in filenames:
                    path = os.path.join(dirname, filename)
                    image_id = filename
                    self.data = self.data.append(
                        {'image_id': image_id,'image_path':path}, ignore_index=True)


class TrainDataset(Dataset):
    def __init__(self, transform=None):
        df = Data(is_train=True).data
        self.file_path = df['image_path'].values
        self.df = df  # .drop(['image_path'], axis=1)
        self.file_name = df['image_id'].values
        self.labels = df['label'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path = f'{self.file_path[idx]}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(224,224))
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx]).long()
        return image, label
    

class TestDataset(Dataset):
    def __init__(self, transform=None):
        df = Data(is_train=False).data
        self.file_path = df['image_path'].values
        self.df = df
        self.file_name = df['image_id'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        file_path = f'{self.file_path[idx]}'
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        if self.transform:
            image = self.transform(image)
        return image
