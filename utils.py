import torch
import logging
import random
import torch.nn as nn
from model import model_cifar
from partition_data import *
import torchvision.models as models
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import random_split

def init_nets(args,n_parties):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in ["cifar10","mnist","fmnist","SVHN"]:
        n_classes = 10
    elif args.dataset in ["cifar100"]:
        n_classes = 100

    for net_i in range(n_parties):
        net = model_cifar(args, n_classes)
        nets[net_i] = net
    return nets


def init_dataloader_per(args, net_dataidx_map_train=None,net_dataidx_map_test=None):
    print("starting init dataloader")
    train_dl_local_set = []
    train_ds_local_set = []
    test_dl_local_set = []
    test_ds_local_set = []

    for i in range(args.n_parties):
        if net_dataidx_map_train==None:
            dataidxs_train=None
            dataidxs_test=None
        else:
            dataidxs_train = net_dataidx_map_train[i]
            dataidxs_test=net_dataidx_map_test[i]
        train_dl_local, test_dl_local, train_ds_local, test_ds_local,test_global = get_dataloader_per(args,dataidxs_train=dataidxs_train,dataidxs_test=dataidxs_test)
        train_dl_local_set.append(train_dl_local)
        train_ds_local_set.append(train_ds_local)
        test_dl_local_set.append(test_dl_local)
        test_ds_local_set.append(test_ds_local)

    print("finishing init dataloader")

    return train_dl_local_set, test_dl_local_set,train_ds_local_set,test_ds_local_set,test_global

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass