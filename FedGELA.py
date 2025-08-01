from load_data import *
import numpy as np
import torch
import argparse
import random
import os
import logging
import datetime
from partition_data import *
import torch.optim as optim
from utils import *
import json
import time

def produce_Ew(train_ds_local_set,num_classes):
    Ew_set=[]
    Mask_set=[]
    for net_id in range(len(train_ds_local_set)):
        train_ds=train_ds_local_set[net_id]
        try:
            
            ds=train_ds_local_set[net_id].dataset
            indices=train_ds_local_set[net_id].indices
            targets=np.array(ds.targets)[indices]
        except:
            targets=train_ds.targets
        
        label=torch.tensor(targets)
        label.requires_grad = False

        uni_label, count = torch.unique(label, return_counts=True)
        num_samples=len(label)

        Ew = torch.zeros(num_classes)
        Mask=torch.zeros(num_classes)
        for idx,class_id in enumerate(uni_label):
            Ew[class_id]=1*count[idx]/num_samples
            Mask[class_id]=1
        Ew=Ew*num_classes
        Ew=Ew.cuda()
        Mask=Mask.cuda()
        Ew_set.append(Ew)
        Mask_set.append(Mask)
    
    return Mask_set,Ew_set

def compute_accuracy_g(model, dataloader,temperature):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0

    criterion = nn.CrossEntropyLoss().cuda()
    loss_collector = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _,_,out = model(x)
            out = torch.matmul(out, ETF)
            loss = criterion(out, target)
            _, pred_label = torch.max(out.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        avg_loss = sum(loss_collector) / len(loss_collector)
    b_acc=balanced_accuracy_score(true_labels_list,pred_labels_list)
    if was_training:
        model.train()

    return correct / float(total), b_acc
def compute_accuracy_l(net_id,model,dataloader):
    was_training = False
    model.eval()

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    sfm=nn.LogSoftmax(dim=1)
    criterion = nn.NLLLoss().cuda()
    loss_collector = []

    with torch.no_grad():
        
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.cuda(), target.to(dtype=torch.int64).cuda()
            _, _,out_g = model(x)
            
            learned_norm = EW_set[net_id]
            cur_M = learned_norm * ETF
            out_g = torch.matmul(out_g, cur_M)
            logits=sfm(out_g)
            loss_global=criterion(logits,target)

            loss=loss_global

            loss = criterion(out_g, target)

            _, pred_label = torch.max(out_g.data, 1)
            loss_collector.append(loss.item())
            total += x.data.size()[0]
            correct += (pred_label == target.data).sum().item()


            pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
            true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())
        
        avg_loss = sum(loss_collector) / len(loss_collector)
        b_acc=balanced_accuracy_score(true_labels_list,pred_labels_list)

    model.train()

    return correct / float(total), b_acc
def get_etf(feat_in,num_classes):
    a=np.random.random(size=(feat_in,num_classes))
    P,_=np.linalg.qr(a)
    P=torch.tensor(P).float()
    I=torch.eye(num_classes)
    one=torch.ones(num_classes,num_classes)
    M=np.sqrt(num_classes/(num_classes-1))*torch.matmul(P,I-((1/num_classes)*one))
    return M

def set_seed(args):
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition', type=str, default="class",
                        help='partition strategies')
    parser.add_argument('--proxy', type=str, default="etf",
                        help='type of proxy')
    parser.add_argument('--n_label', type=int, default=3,
                        help='num of label of each client')
    parser.add_argument('--temperature', type=float, default=0.2,
                        help='temperature of feature')
    parser.add_argument('--mu', type=float, default=0.01,
                        help='param of baselines')
    parser.add_argument('--logdir', type=str, required=False, default="./log/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--train_batchsize', type=int, default=64, help='batchsize for training')
    parser.add_argument('--test_batchsize', type=int, default=100, help='batchsize for testing')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--model', type=str, default='resnet18', help='neural network used in training')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--party_per_round', type=int, default=10, help='how many clients are sampled in each round')
    parser.add_argument('--comm_round', type=int, default=400, help='number of maximum communication roun')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    args = parser.parse_args()
    return args

def train_net(net_id, net,train_dataloader, test_dataloader, epochs, lr, optimizer,reg,temperature):
    
    net.cuda()
    best_testacc=0
    test_acc,  _ = compute_accuracy_l(net_id,net, test_dataloader)
    if test_acc>best_testacc:
        best_testacc=test_acc
    net.train()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    optimizer = optim.SGD([{"params":net.parameters()}], lr=lr, momentum=0.9,
                              weight_decay=reg)
    criterion = nn.NLLLoss().cuda()
    sfm=nn.LogSoftmax(dim=1)

    for epoch in range(epochs):
        epoch_loss_collector = []
        time_start = time.time()
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            _, _,out_g = net(x)

            learned_norm = EW_set[net_id]
            cur_M = learned_norm * ETF
            out_g = torch.matmul(out_g, cur_M)
            
            logits=sfm(out_g/temperature)
            loss_global=criterion(logits,target)
            loss=loss_global
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),max_norm=20,norm_type=2)
            optimizer.step()

            epoch_loss_collector.append(loss.item())

        time_end = time.time()

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        test_acc,  bacc = compute_accuracy_l(net_id,net, test_dataloader)
        if test_acc>best_testacc:
            best_testacc=test_acc


    logger.info(">> Net id:"+str(net_id)+" Test accuracy: %f" % best_testacc)
    logger.info(">> Net id:"+str(net_id)+" Balanced Test accuracy: %f" % bacc)
    net.to('cpu')

    logger.info(' ** Training complete **')
    return 0, test_acc


def local_train_net(nets, selected, args, train_dl_set, test_dl_set):

    for net_id, net in nets.items():
        if net_id in selected:
            logger.info("Training network %s" % (str(net_id)))
            trainacc, testacc = train_net(net_id, net, train_dl_set[net_id], test_dl_set[net_id], args.epochs, args.lr,
                                          args.optimizer, args.reg,args.temperature)
            logger.info("net %d final test acc %f" % (net_id, testacc))

    return nets


if __name__ == '__main__':

    args = get_args()
    mkdirs(args.logdir)

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))

    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    set_seed(args)
    
    logger.info("Partitioning data")
    if args.dataset in ["cifar10","cifar100","tiny-imagenet","mnist","fmnist","SVHN"]:

        if args.dataset in ["cifar10","mnist","fmnist","SVHN"]:
            n_class=10
        elif args.dataset=="cifar100":
            n_class=100
        elif args.dataset == "tiny-imagenet":
            n_class=200
    if args.partition=="class":
        if args.dataset in ["cifar10","cifar100","tiny-imagenet","mnist","fmnist","SVHN"]:

            if args.dataset in ["cifar10","mnist","fmnist","SVHN"]:
                n_class=10
            elif args.dataset=="cifar100":
                n_class=100
            elif args.dataset == "tiny-imagenet":
                n_class=200
            net_dataidx_map_train,net_dataidx_map_test = partition_class_per(args.dataset,
                                                            args.n_parties,
                                                            n_class,
                                                            args.n_label)

            train_dl_local_set, test_dl_set, train_ds_local_set, test_ds_set,test_global = init_dataloader_per(args,net_dataidx_map_train,net_dataidx_map_test)
        elif args.dataset in ["femnist_by_writer", "synthetic", "shakespeare", "PACS", "officehome","ISIC"]:
            train_dl_local_set, test_dl_set, train_ds_local_set, test_ds = init_dataloader_per(args)
    elif args.partition=="dirichlet":
        net_dataidx_map_train, net_dataidx_map_test = partition_diri_per(args.dataset,
                                                            args.n_parties,
                                                            beta=args.beta)

        train_dl_local_set, test_dl_set, train_ds_local_set, test_ds_set,test_global = init_dataloader_per(args,net_dataidx_map_train,net_dataidx_map_test)

    global ETF
    global EW_set
    global MASK_set
    if args.dataset=="cifar10":
        ETF=get_etf(84,n_class).cuda()
    if args.dataset=="SVHN":
        ETF=get_etf(84,n_class).cuda()
    if args.dataset=="fmnist":
        ETF=get_etf(84,n_class).cuda()
    elif args.dataset=="cifar100":
        ETF=get_etf(512,n_class).cuda()
    MASK_set,EW_set=produce_Ew(train_ds_local_set,n_class)

    logger.info("recording params")
    for key in vars(args):
        value=vars(args)[key]
        logger.info(str(key)+":"+str(value))
    
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, args.party_per_round))

    train_dl = None
    logger.info("Initializing nets")
    nets = init_nets(args,args.n_parties)
    global_models = init_nets(args,1)
    global_model = global_models[0]
    global_para = global_model.state_dict()

    acc=0
    for net_id, net in nets.items():
        net.load_state_dict(global_para)
    n_comm_rounds = args.comm_round

    if args.load_model_file :
        global_model.load_state_dict(torch.load(args.load_model_file))
        n_comm_rounds -= args.load_model_round

    weights = []
    for i in range(args.party_per_round):
        weights.append([])
        weights[i] = 0
    logger.info("start training")


    for round in range(args.comm_round):
        logger.info("in comm round:" + str(round))

        arr = np.arange(args.n_parties)
        np.random.shuffle(arr)
        selected = arr[:args.party_per_round]
        local_data_points = [len(train_dl_local_set[r]) for r in selected]

        index = 0
        for i in range(len(local_data_points)):
            weights[i] += local_data_points[i]

        global_para = global_model.state_dict()

        local_train_net(nets,selected, args, train_dl_set=train_dl_local_set, test_dl_set=test_dl_set)
        print("updating global")

        for idx in range(len(selected)):
            net_id=selected[idx]
            net_para = nets[net_id].cpu().state_dict()
            weight = weights[idx] / sum(weights)
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * weight
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * weight
            nets[net_id]=nets[net_id]
        global_model.load_state_dict(global_para)


        logger.info('global n_test: %d' % len(test_global))
        global_model.cuda()
        
        test_acc, bacc, = compute_accuracy_g(global_model, test_global,args.temperature)
        
        global_model.to('cpu')
        if test_acc>acc:
            acc=test_acc
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Balanced Test accuracy: %f' % bacc)
        for net_id, net in nets.items():
            net.load_state_dict(global_para)
        for idx in range(int(args.party_per_round)):
            weights[idx] = 0
