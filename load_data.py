import torchvision.transforms as transforms
from dataset import *
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader


def load_cifar10_data():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = CIFAR10(train=True,  transform=transform)
    cifar10_test_ds = CIFAR10(train=False,  transform=transform)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets
    return (X_train, y_train, X_test, y_test)

def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(train=True, transform=transform)
    mnist_test_ds = MNIST_truncated(train=False, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_svhn_data():

    transform = transforms.Compose([transforms.ToTensor()])

    svhn_train_ds = SVHN_custom( train=True, transform=transform)
    svhn_test_ds = SVHN_custom( train=False, transform=transform)

    X_train, y_train = svhn_train_ds.data, svhn_train_ds.targets
    X_test, y_test = svhn_test_ds.data, svhn_test_ds.targets

    return (X_train, y_train, X_test, y_test)


def load_fmnist_data():
    transform = transforms.Compose([transforms.ToTensor()])

    fmnist_train_ds = FashionMNIST_truncated(train=True, transform=transform)
    fmnist_test_ds = FashionMNIST_truncated(train=False, transform=transform)

    X_train, y_train = fmnist_train_ds.data, fmnist_train_ds.targets
    X_test, y_test = fmnist_test_ds.data, fmnist_test_ds.targets

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)
def load_tiny_imagenet_data():
    transform = transforms.Compose([transforms.ToTensor()])
    tinyimagenet_train_ds = tinyimagenet_s3(train=True,  transform=transform)
    tinyimagenet_test_ds = tinyimagenet_s3(train=False,  transform=transform)
    X_train, y_train = tinyimagenet_train_ds.data, tinyimagenet_train_ds.targets
    X_test, y_test = tinyimagenet_test_ds.data, tinyimagenet_test_ds.targets
    return (X_train, y_train, X_test, y_test)

def load_cifar100_data():
    transform = transforms.Compose([transforms.ToTensor()])
    cifar100_train_ds = CIFAR100(train=True,  transform=transform)
    cifar100_test_ds = CIFAR100(train=False,  transform=transform)
    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.targets
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.targets
    return (X_train, y_train, X_test, y_test)



def get_dataloader_per(args, dataidxs_train=None,dataidxs_test=None,identity=None, noise_level=0):
    if args.dataset in ('cifar10', 'cifar100'):
        if args.dataset == 'cifar10':
            dl_obj = CIFAR10
            normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        elif args.dataset == 'cifar100':
            dl_obj = CIFAR100
            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])

        train_ds = dl_obj(dataidxs=dataidxs_train, train=True, transform=transform_train)
        test_ds = dl_obj(dataidxs=dataidxs_test,train=False, transform=transform_test)
        test_global_ds=dl_obj(dataidxs=None,train=False,transform=transform_test)
    elif args.dataset=="mnist":

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        train_ds = MNIST_truncated(dataidxs=dataidxs, train=True, transform=transform_train)
        test_ds = MNIST_truncated(train=False, transform=transform_test)
    elif args.dataset=="fmnist":

        transform_train = transforms.Compose([
            transforms.ToTensor(),

            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),

            ])
        train_ds = FashionMNIST_truncated(dataidxs=dataidxs_train, train=True, transform=transform_train)
        test_ds = FashionMNIST_truncated(dataidxs=dataidxs_test,train=False, transform=transform_test)
        test_global_ds=FashionMNIST_truncated(dataidxs=None,train=False,transform=transform_test)
        

    elif args.dataset=="SVHN":

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614),
            ),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614),
            ),
            ])
        train_ds = SVHN_custom(dataidxs=dataidxs_train, train=True, transform=transform_train)
        test_ds = SVHN_custom(dataidxs=dataidxs_test,train=False, transform=transform_test)
        test_global_ds=SVHN_custom(dataidxs=None,train=False,transform=transform_test)

    elif args.dataset=="tiny-imagenet":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        train_ds = tinyimagenet_s3(dataidxs=dataidxs, train=True, transform=transform_train)
        test_ds = tinyimagenet_s3(train=False, transform=transform_test)
    
    print("finishing init dataloader")
    train_dl = DataLoader(dataset=train_ds, batch_size=args.train_batchsize, drop_last=True, shuffle=True,num_workers=4)
    test_dl = DataLoader(dataset=test_ds, batch_size=args.test_batchsize, shuffle=False, drop_last=False,num_workers=4)
    test_global=DataLoader(dataset=test_global_ds, batch_size=args.test_batchsize, shuffle=False, drop_last=False,num_workers=4)
    return train_dl, test_dl, train_ds, test_ds,test_global