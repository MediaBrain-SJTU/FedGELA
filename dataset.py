import os.path
import albumentations
import pickle
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import torch.utils.data as data
from PIL import Image
import numpy as np

import os.path
import torchvision
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity

import os
import os.path
import pandas as pd
import torch
from typing import Any, Callable, Optional, Tuple

class CIFAR10(data.Dataset):
    def __init__(self,url="./data/cifar-10-batches-py/", train=True,transform=None, target_transform=None,dataidxs=None):
        self.dataidxs=dataidxs
        self.transform=transform
        self.target_transform=target_transform
        self.url=url
        self.train_list = ["data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5"]
        self.test_list = ["test_batch"]
        self.data=[]
        self.targets=[]
        if train:
            for file in self.train_list:
                path=self.url+file
                with open(path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        else:
            for file in self.test_list:
                path=self.url+file
                with open(path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self.targets=np.array(self.targets)
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self) -> int:
        return len(self.data)


class CIFAR100(data.Dataset):
    def __init__(self,url="./data/cifar100/", train=True,transform=None, target_transform=None,dataidxs=None):
        self.dataidxs = dataidxs
        self.transform=transform
        self.target_transform=target_transform
        self.url=url
        self.train_list = ["train"]
        self.test_list = ["test"]
        self.data=[]
        self.targets=[]
        if train:
            for file in self.train_list:
                path=self.url+file
                with open(path, "rb") as f:

                    entry = pickle.load(f, encoding="latin1")
                
                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])
        else:
            for file in self.test_list:
                path=self.url+file
                with open(path, "rb") as f:
                    entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  
        self.targets = np.array(self.targets)
        if self.dataidxs is not None:
            self.data = self.data[self.dataidxs]
            self.targets = self.targets[self.dataidxs]
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    def __len__(self) -> int:
        return len(self.data)

class FashionMNIST_truncated(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = FashionMNIST("./data/fmnist/", self.train, self.transform, self.target_transform, True)
        data = mnist_dataobj.data
        targets = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):

        img, targets = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)

class MNIST_truncated(data.Dataset):

    def __init__(self,  dataidxs=None, train=True, transform=None, target_transform=None):


        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        mnist_dataobj = MNIST("./data/mnist/", self.train, self.transform, self.target_transform, True)
        data = mnist_dataobj.data
        target = mnist_dataobj.targets

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

class SVHN_custom(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None):


        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.data, self.targets = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if self.train is True:

            svhn_dataobj = SVHN("./data/SVHN/", 'train', self.transform, self.target_transform,True)
            data = svhn_dataobj.data
            targets = svhn_dataobj.labels
        else:
            svhn_dataobj = SVHN("./data/SVHN/", 'test', self.transform, self.target_transform,True)
            data = svhn_dataobj.data
            targets = svhn_dataobj.labels

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            targets = targets[self.dataidxs]

        return data, targets

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)
