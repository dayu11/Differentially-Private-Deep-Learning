import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models import *


import numpy as np
import random

import time

from get_noise_variance import get_sigma


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--arch', default='resnet28', type=str, help='model name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet', type=str, help='session name')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
parser.add_argument('--n_epoch', default=400, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.4, type=float, help='base learning rate (default=0.4)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum coeeficient')

parser.add_argument('--eps', default=8, type=float, help='eps value')
parser.add_argument('--width', default=10, type=int, help='model width')
parser.add_argument('--delta', default=1e-5, type=float, help='delta value')
parser.add_argument('--rank', default=16, type=int, help='rank of reparameterization')
parser.add_argument('--clipping', default=1., type=float, help='clipping threshold')
parser.add_argument('--warmup_epoch', default=-1, type=int, help='num. of epochs for warmup')


args = parser.parse_args()

use_cuda = torch.cuda.is_available()
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize



# Data
print('==> Preparing data..')
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


dataset_func = torchvision.datasets.CIFAR10

trainset = dataset_func(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = dataset_func(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=2)



# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_file = './checkpoint/' + args.sess + '.ckpt'
    checkpoint = torch.load(checkpoint_file)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    print('resume succeed')
else:
    print("=> creating model '{}'".format(args.arch))
    out_dim = 10
    
    hyper_params_dict = {}
    hyper_params_dict['rank'] = args.rank
    hyper_params_dict['width'] = args.width

    net = eval(args.arch+'(out_dim, hyper_params_dict)')
    print('resume failed')



datasize = 50000
q = args.batchsize / datasize
steps = int(1/q) * args.n_epoch
sigma = get_sigma(q, steps, args.eps, args.delta)[0]
print('noise standard deviation for eps = %.1f: '%args.eps, sigma)


result_folder = './results/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
logname = result_folder +  args.sess  + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    num_gpu = torch.cuda.device_count()
    print('Using', num_gpu, 'GPUs.')
    cudnn.benchmark = True
    if(num_gpu == 1):
        print('Using CUDA..')
    else:
        print('Do not support parallelism, please use only one GPU. (try \'CUDA_VISIBLE_DEVICES=0 python .... \')')


loss_func = nn.CrossEntropyLoss()


params = []
# list of low-rank parameters
ghost_params = []

for p in net.named_parameters():
    # we do not reparametrize linear layer because it is already low-rank
    if('full' in p[0] or 'fc' in p[0]):
        params.append(p[1])
        p[1].requires_grad = False
    elif('left' in p[0]):
        ghost_params.append(p[1])

for p in params:
    p.cached_grad = None

# add the parameter of linear layer to low-rank parameters
ghost_params += [params[-1]]


# we use this optimizer to use the gradient reconstructed from the gradient carriers
optimizer = optim.SGD(
    params,
    lr=args.lr, 
    momentum=args.momentum, 
    weight_decay=args.weight_decay)


# dummy optimizer, we use this optimizer to clear the gradients of gradient carriers
ghost_optimizer = optim.SGD(ghost_params, lr=1)


num_p = 0
for p in ghost_params:
    num_p += p.numel()

print('number of parameters (low-rank): %.3f M'%(num_p/1000000), end=' ')

num_p = 0
for p in params:
    num_p += p.numel()

print('number of parameters (full): %.3f M'%(num_p/1000000))

def clip_column(grad_sample, threshold=1.0):
    norms = torch.norm(grad_sample.view(grad_sample.shape[0], -1), dim=1)
    scale = torch.clamp(threshold/norms, max=1.0)
    grad_sample *= scale.view(-1, 1, 1)

def process_grad_sample(params, clipping=1, inner_t=0):
    n = params[0].grad_sample.shape[0]
    grad_norm_list = torch.zeros(n).cuda()
    for p in params: 
        flat_g = p.grad_sample.view(n, -1)
        current_norm_list = torch.norm(flat_g, dim=1)
        grad_norm_list += torch.square(current_norm_list)
    grad_norm_list = torch.sqrt(grad_norm_list)
    scaling = clipping/grad_norm_list
    scaling[scaling>1] = 1

    for p in params:
        p_dim = len(p.shape)
        scaling = scaling.view([n] + [1]*p_dim)
        p.grad_sample *= scaling
        if(inner_t == 0):
            p.grad = torch.sum(p.grad_sample, dim=0)
        else:
            p.grad += torch.sum(p.grad_sample, dim=0)
        p.grad_sample.mul_(0.)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()


        for m in net.modules():
            if(hasattr(m, '_update_weight')):
                m.is_training = True
        with torch.no_grad():
            net.module.decomposite_weight()

        # use multiple micro-batches
        stepsize = 250
        inner_t = args.batchsize // stepsize
        if(args.batchsize % stepsize != 0):
            raise 'batchsize should be an integer multiple of 250.'
        
        loss = None
        outputs_list = []
        for t in range(inner_t):
            tiny_inputs, tiny_targets = inputs[t*stepsize:(t+1)*stepsize], targets[t*stepsize:(t+1)*stepsize]
            tiny_outputs = net(tiny_inputs)
            tiny_loss = loss_func(tiny_outputs, tiny_targets)
            tiny_loss.backward()
            # gradient clipping
            process_grad_sample(ghost_params, clipping=args.clipping, inner_t=t)
            if(loss == None):
                loss = tiny_loss.detach()/inner_t
            else:
                loss += tiny_loss.detach()/inner_t
            outputs_list.append(tiny_outputs.detach())

        # add noise for DP
        for p in ghost_params:
            p.grad /= args.batchsize
            p.grad += torch.normal(0, sigma*args.clipping/args.batchsize, size = p.shape).cuda()
        # reconstruct update
        with torch.no_grad():
            for module in net.modules():
                if(hasattr(module, 'get_full_grad')):
                    full_grad = module.get_full_grad(args.lr)
                    module.full_conv.weight.grad = full_grad

        net.module.update_weight()
        outputs = torch.cat(outputs_list)
        optimizer.step()
        optimizer.zero_grad()
        ghost_optimizer.zero_grad()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    
    if(epoch + 1 == args.warmup_epoch):
        # take a snapshot of current model for computing historical update
        net.module.update_init_weight()


    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')

    return (train_loss/batch_idx, acc)

def test(epoch):
    net.eval()
    for m in net.modules():
        if(hasattr(m, '_update_weight')):
            m.is_training = False
    test_loss = 0
    correct = 0
    total = 0


    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)


            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()


        acc = 100.*float(correct)/float(total)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)


    return (test_loss/batch_idx, acc)

def checkpoint(acc, epoch):
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + args.sess + '.ckpt')

def adjust_learning_rate(optimizer, epoch):
    if(epoch<0.3*args.n_epoch):
        decay = 1.
    elif(epoch<0.6*args.n_epoch):
        decay = 5.
    elif(epoch<0.8*args.n_epoch):
        decay = 25.
    else:
        decay = 125.
    

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr / decay
    return args.lr / decay

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])


for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, epoch)

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, lr, train_loss, train_acc, test_loss, test_acc])
