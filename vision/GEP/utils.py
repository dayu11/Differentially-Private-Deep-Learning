
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import os


import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent

def get_data_loader(dataset, batchsize):
    if(dataset == 'svhn'):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.SVHN('./data',split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=73257, shuffle=True, num_workers=0) #load full btach into memory, to concatenate with extra data

        extraset = torchvision.datasets.SVHN('./data',split='extra', download=True, transform=transform)
        extraloader = torch.utils.data.DataLoader(extraset, batch_size=531131, shuffle=True, num_workers=0) #load full btach into memory

        testset = torchvision.datasets.SVHN('./data',split='test', download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        return trainloader, extraloader, testloader, len(trainset)+len(extraset), len(testset)
    else:
        transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train) 
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test) 
        testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)
        return trainloader, testloader, len(trainset), len(testset)



def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32, rgp=True):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T
        if(rgp):
            rdp = compute_rdp(q, cur_sigma, steps, orders) * 2 ## when using residual gradients, the sensitivity is sqrt(2)
        else:
            rdp = compute_rdp(q, cur_sigma, steps, orders)
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
## rgp: use residual gradient perturbation or not
def get_sigma(q, T, eps, delta, init_sigma=10, interval=1., rgp=True):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    interval /= 10
    cur_sigma, previous_eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rgp=rgp)
    return cur_sigma, previous_eps


def restore_param(cur_state, state_dict):
    own_state = cur_state
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        own_state[name].copy_(param)

def sum_list_tensor(tensor_list, dim=0):
    return torch.sum(torch.cat(tensor_list, dim=dim), dim=dim)

def flatten_tensor(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i] = tensor_list[i].reshape([tensor_list[i].shape[0], -1])
    flatten_param = torch.cat(tensor_list, dim=1)
    del tensor_list
    return flatten_param


def checkpoint(net, acc, epoch, sess):
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'approx_error': net.gep.approx_error
    }
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + sess  + '.ckpt')

def adjust_learning_rate(optimizer, init_lr, epoch, all_epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    decay = 1.0
    if(epoch<all_epoch*0.5):
        decay = 1.
    elif(epoch<all_epoch*0.75):
        decay = 10.
    else:
        decay = 100.

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / decay
    return init_lr / decay
