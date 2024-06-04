from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
from resnetcifar import ResNet18_cifar10, ResNet50_cifar10,ResNet18_mnist,ResNet18_cifar10_align
from resnet import *
import torch
import torchvision.models as models
from collections import OrderedDict
import torchvision.transforms as transforms
from get_proxy import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
class shake_backbone(nn.Module):
    def __init__(self):
        super(shake_backbone, self).__init__()
        self.embed = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
    def forward(self, x):
        x = self.embed(x)

        x, hidden = self.lstm(x)
        x=x[:, -1, :]
        #print(x.shape)
        return x

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class Classifier(nn.Module):
    def __init__(self, feat_in, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(feat_in, num_classes)

        self.apply(_weights_init)
        self.fc.weight.requires_grad = False
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x




def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])

def resnet32_fe():
    return ResNet_fe(BasicBlock, [5, 5, 5])

def resnet32_fe_c100():
    print('resnet32 arch for cifar 100 (32 in channels)')
    return ResNet_fe(BasicBlock, [5,5,5], 32)

def resnet32(num_classes=10, use_norm=False):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])

def reg_ETF(output, label, classifier, mse_loss):
#    cur_M = classifier.cur_M
    target = classifier.cur_M[:, label].T  ## B, d
    loss = mse_loss(output, target)
    return loss

def dot_loss(output, label, cur_M, classifier, criterion, H_length, reg_lam=0):
    target = cur_M[:, label].T ## B, d  output: B, d
    if criterion == 'dot_loss':
        loss = - torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1).mean()
    elif criterion == 'reg_dot_loss':
        dot = torch.bmm(output.unsqueeze(1), target.unsqueeze(2)).view(-1) #+ classifier.module.bias[label].view(-1)

        with torch.no_grad():
            M_length = torch.sqrt(torch.sum(target ** 2, dim=1, keepdims=False))
        loss = (1/2) * torch.mean(((dot-(M_length * H_length)) ** 2) / H_length)

        if reg_lam > 0:
            reg_Eh_l2 = torch.mean(torch.sqrt(torch.sum(output ** 2, dim=1, keepdims=True)))
            loss = loss + reg_Eh_l2*reg_lam

    return loss

def produce_Ew(label, num_classes):
    uni_label, count = torch.unique(label, return_counts=True)
    batch_size = label.size(0)
    uni_label_num = uni_label.size(0)
    assert batch_size == torch.sum(count)
    gamma = batch_size / uni_label_num
    Ew = torch.ones(1, num_classes).cuda(label.device)
    for i in range(uni_label_num):
        label_id = uni_label[i]
        label_count = count[i]
        length = torch.sqrt(gamma / label_count)
#        length = (gamma / label_count)
        #length = torch.sqrt(label_count / gamma)
        Ew[0, label_id] = length
    return Ew

def produce_global_Ew(cls_num_list):
    num_classes = len(cls_num_list)
    cls_num_list = torch.tensor(cls_num_list).cuda()
    total_num = torch.sum(cls_num_list)
    gamma = total_num / num_classes
    Ew = torch.sqrt(gamma / cls_num_list)
    Ew = Ew.unsqueeze(0)
    return Ew

class etf_classifier(nn.Module):
    def __init__(self,feat_in,num_classes,fix_bn=False,LWS=False,reg_ETF=False):
        super(etf_classifier,self).__init__()
        P=self.generate_random_orthogonal_matrix(feat_in,num_classes)
        I=torch.eye(num_classes)
        one=torch.ones(num_classes,num_classes)
        M=np.sqrt(num_classes/(num_classes-1))*torch.matmul(P,I-((1/num_classes)*one))
        self.ori_M=M.cuda()
        self.LWS=LWS
        self.reg_ETF=reg_ETF
        self.BN_H=nn.BatchNorm1d(feat_in)
        if fix_bn:
            self.BN_H.weight.requires_grad=False
            self.BN_H.bias.requires_grad=False
    
    def generate_random_orthogonal_matrix(self,feat_in,num_classes):
        a=np.random.random(size=(feat_in,num_classes))
        P,_=np.linalg.qr(a)
        P=torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T,P),torch.eye(num_classes),atol=1e-07),torch.max(torch.abs(torch.matmul(P.T,P)-torch.eye(num_classes)))
        return P
    def forward(self,x):
        x=self.BN_H(x)
        return X


class orthogonal(nn.Module):
    def __init__(self,feat_dim=128,class_num=10):
        super(orthogonal,self).__init__()
        
        #self.proxy=nn.Linear(feat_dim,class_num)
        #elif type=="orthogonal":
        proxy_dim=np.ceil(class_num / 2).astype(int)
        self.encode=nn.Linear(feat_dim,proxy_dim)

        # if feat_dim != np.ceil(num_classes / 2).astype(int):
        #     raise ValueError("wrong number of feat_dim")
        #self.
        vec = np.identity(proxy_dim)
        vec = np.vstack((vec, -vec))
        #vec
        vec=torch.tensor(vec,dtype=torch.float)
        #print(vec.shape)
        self.proxy=nn.Parameter(vec)
        self.proxy.requires_grad=False
        #self.proxy.requires_grad=False
    def forward(self,feature):
        hidden=self.encode(feature)
        #print(hidden.shape)
        output=torch.mm(hidden,self.proxy.T)
        return output

def no_meaning(x):
    return x

def l2norm(x):
    x=nn.functional.normalize(x,p=2,dim=1)
    return x

class l2noetf(nn.Module):
    def __init__(self,feat_dim,class_num):
        super(l2noetf,self).__init__()
        self.model=nn.Linear(feat_dim,class_num)
    def forward(self,x):
            #nn.Linear(feat_dim,class_num)
        x=nn.functional.normalize(x,p=2,dim=1)
        x=self.model(x)
        return x

class l2CLS(nn.Module):
    def __init__(self,feat_dim,class_num):
        super(l2CLS,self).__init__()
        self.model=nn.Linear(feat_dim,class_num)
    def forward(self,x):
            #nn.Linear(feat_dim,class_num)
        x=nn.functional.normalize(x,p=2,dim=1)
        x=self.model(x)
        return x
class proxies(nn.Module):
    def __init__(self,class_num=10,feat_dim=10,type="orthogonal"):
        super(proxies,self).__init__()
        if type=="cls":
            self.proxy=nn.Linear(feat_dim,class_num)
        elif type=="cls_norm":
            self.proxy=nn.Sequential(nn.BatchNorm1d(feat_dim),
            nn.Linear(feat_dim,class_num),
            )
        elif type=="orthogonal":
            self.proxy=orthogonal(feat_dim=feat_dim,class_num=class_num)
        elif type =="etf":
            self.proxy=nn.BatchNorm1d(feat_dim)
        elif type=="etf_p":
            self.proxy=nn.BatchNorm1d(feat_dim)
        elif type =="etfp_nonorm":
            self.proxy=no_meaning
        elif type=="l2":
            self.proxy=l2norm
        elif type=="l2noetf":
            self.proxy=l2noetf
        elif type=="l2norm":
            self.proxy=nn.Sequential(
                nn.BatchNorm1d(feat_dim),
                l2norm
            )
        elif type=="clsl2":
            self.proxy=l2CLS(feat_dim,class_num)

    def forward(self,feature):
        return self.proxy(feature)
class SimpleCNN_header(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10):
        super(SimpleCNN_header, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        # x = self.fc3(x)
        return x



class model_cifar(nn.Module):
    def __init__(self, args , n_classes,out_dim=256):
        super(model_cifar, self).__init__()

        if args.model == "resnet50":
            basemodel = ResNet50_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
        elif args.model=="resnet18_align":   
            basemodel = ResNet18_cifar10_align()
        elif args.model=="resnet18":
            if args.dataset in ["mnist","fmnist"]:
                basemodel=ResNet18_mnist()
            
            else:
                #basemodel =resnet32_fe()
                #num_ftrs = 64
                basemodel = ResNet18_cifar10()
            self.backbone = nn.Sequential(*list(basemodel.children())[:-1])
            num_ftrs = basemodel.fc.in_features
            #num_ftrs = 84
        elif args.model == 'simple-cnn':
            self.backbone = SimpleCNN_header(input_dim=(16 * 5 * 5), hidden_dims=[120, 84])
            num_ftrs = 84
        elif args.model=="efficient":
            #self.backbone=EfficientNet.from_pretrained('efficientnet-b0')
            self.backbone=EfficientNet.from_name('efficientnet-b0')
            num_ftrs = self.backbone._fc.in_features
            #num_ftrs = 1000
            self.backbone._fc = nn.Identity()
        elif args.model=="cifar_pacs":
            backbone = models.resnet18(num_classes=2048, pretrained=False)
            param_dicts = torch.load('/mnt/workspace/colla_group/ckpt/resnet18-5c106cde.pth')

            for item in list(param_dicts):
                if "fc" in item:
                    del param_dicts[item]
            backbone.load_state_dict(param_dicts, strict=False)
            self.backbone = backbone
            num_ftrs=2048
        elif args.model=="shake_lstm":
            self.backbone=shake_backbone()
            num_ftrs=256
        proxy_type=args.proxy
        if args.dataset=="shakespeare":
            out_dim=256
        if args.dataset in ["PACS","officehome"]:
            out_dim=84
        if args.dataset=="ISIC":
            out_dim=84
        if args.dataset=="cifar10":
            out_dim=512
        if args.dataset=="SVHN":
            #out_dim=84
            out_dim=512
        if args.dataset=="fmnist":
            out_dim=84
        elif args.dataset=="cifar100":
            #out_dim=84
            out_dim=512
        
        self.encoder=nn.Sequential(

            
            nn.Linear(num_ftrs, num_ftrs),
            nn.ReLU(),
            nn.Linear(num_ftrs, out_dim),
            
            )

        self.proxy=proxies(n_classes,out_dim,type=proxy_type)


    def forward(self, x):
        h = self.backbone(x)
        h=h.squeeze()
        x=h
        x=self.encoder(x)
        y = self.proxy(x)
        return h, x, y
    def forward_proxy(self, x):

        y = self.proxy(x)
        return y