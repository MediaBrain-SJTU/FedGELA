from load_data import *
import logging
import random
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)
def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list=[]
    for net_id, data in net_cls_counts.items():
        n_total=0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print('mean:', np.mean(data_list))
    print('std:', np.std(data_list))
    logger.info('Data statistics: %s' % str(net_cls_counts))

    return net_cls_counts
def partition_class_per(dataset,n_parties,n_class,n_label=3):
    if n_class==n_label:
        (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)=partition_dirichlet(dataset, n_parties, beta=100000)
    else:
        if dataset =="cifar10":
            X_train, y_train, X_test, y_test = load_cifar10_data()
        elif dataset == "cifar10l":
            X_train, y_train, X_test, y_test = load_cifar10_data()
        elif dataset == 'cifar100':
            X_train, y_train, X_test, y_test = load_cifar100_data()
        elif dataset=="tiny-imagenet":
            X_train, y_train, X_test, y_test = load_tiny_imagenet_data()
        elif dataset=="mnist":
            X_train, y_train, X_test, y_test = load_mnist_data()
        elif dataset=="fmnist":
            X_train, y_train, X_test, y_test = load_fmnist_data()
        elif dataset=="SVHN":
            X_train, y_train, X_test, y_test = load_svhn_data()
        times=[0 for i in range(n_class)]
        contain=[]
        count=0

        for i in range(n_parties):
            contain.append([])

            for j in range(n_label):
                if count < n_class:
                    label_id=count
                    count+=1
                else:
                    while (True):
                        label_id = random.randint(0, n_class - 1)
                        if label_id not in contain[i]:
                            break
                times[label_id] += 1
                contain[i].append(label_id)
        net_dataidx_map_train ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}
        net_dataidx_map_test ={i:np.ndarray(0,dtype=np.int64) for i in range(n_parties)}

        for i in range(n_class):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map_train[j]=np.append(net_dataidx_map_train[j],split[ids])
                    ids+=1

        for i in range(n_class):
            idx_k = np.where(y_test==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map_test[j]=np.append(net_dataidx_map_test[j],split[ids])
                    ids+=1

        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map_train)
        testdata_cls_counts = record_net_data_stats(y_test, net_dataidx_map_test)
    return net_dataidx_map_train,net_dataidx_map_test

def partition_diri_per(dataset,n_parties,beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data()
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data()
    elif dataset == "tiny-imagenet":
        X_train, y_train, X_test, y_test = load_tiny_imagenet_data()
    elif dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data()
    elif dataset == "fmnist":
        X_train, y_train, X_test, y_test = load_fmnist_data()
    elif dataset == "SVHN":
        X_train, y_train, X_test, y_test = load_svhn_data()
    min_size = 0
    min_require_size = 10
    K = 10
    if dataset == 'cifar100':
        K = 100
    if dataset=='tiny-imagenet':
        K=200
    N_train = y_train.shape[0]
    N_test = y_test.shape[0]
    #print(N_train,N_test)
    net_dataidx_map_train = {}
    net_dataidx_map_test = {}
    #while min_size < min_require_size:
    idx_batch_test = [[] for _ in range(n_parties)]
    idx_batch_train = [[] for _ in range(n_parties)]
    for k in range(K):
        idx_k_train = np.where(y_train == k)[0]
        idx_k_test = np.where(y_test == k)[0]
        np.random.shuffle(idx_k_train)
        np.random.shuffle(idx_k_test)
        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        #print(proportions)
        proportions_train = np.array([p * (len(idx_j) < N_train / n_parties) for p, idx_j in zip(proportions, idx_batch_train)])
        proportions_train = proportions_train / proportions_train.sum()
        proportions_train = (np.cumsum(proportions_train) * len(idx_k_train)).astype(int)[:-1]
        idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
        
        proportions_test = np.array([p * (len(idx_j) < N_test / n_parties) for p, idx_j in zip(proportions, idx_batch_test)])
        proportions_test = proportions_test / proportions_test.sum()
        proportions_test = (np.cumsum(proportions_test) * len(idx_k_test)).astype(int)[:-1]
        idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]

    for j in range(n_parties):
        np.random.shuffle(idx_batch_train[j])
        net_dataidx_map_train[j] = idx_batch_train[j]
    #print(idx_batch_test)
    for j in range(n_parties):
        np.random.shuffle(idx_batch_test[j])
        net_dataidx_map_test[j] = idx_batch_test[j]
    
    _ = record_net_data_stats(y_train, net_dataidx_map_train)
    _ = record_net_data_stats(y_test, net_dataidx_map_test)
    return net_dataidx_map_train,net_dataidx_map_test




def partition_dirichlet(dataset,n_parties, beta=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data()
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data()
    elif dataset == "tiny-imagenet":
        X_train, y_train, X_test, y_test = load_tiny_imagenet_data()
    elif dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist_data()
    elif dataset == "fmnist":
        X_train, y_train, X_test, y_test = load_fmnist_data()
    elif dataset == "SVHN":
        X_train, y_train, X_test, y_test = load_svhn_data()
    min_size = 0
    min_require_size = 10
    K = 10
    if dataset == 'cifar100':
        K = 100
    if dataset=='tiny-imagenet':
        K=200
    N = y_train.shape[0]
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])


    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)

