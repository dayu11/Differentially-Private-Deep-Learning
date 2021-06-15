import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import numpy as np

from models import resnet20, GEP
from utils import get_data_loader, get_sigma, restore_param, sum_list_tensor, flatten_tensor, checkpoint, adjust_learning_rate

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

parser = argparse.ArgumentParser(description='Differentially Private learning with GEP')

## general arguments
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=2, type=int, help='random seed')
parser.add_argument('--weight_decay', default=2e-4, type=float, help='weight decay')
parser.add_argument('--batchsize', default=1000, type=int, help='batch size')
parser.add_argument('--n_epoch', default=200, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--eps', default=8., type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

parser.add_argument('--rgp', action='store_true', help='use residual gradient perturbation or not')
parser.add_argument('--clip0', default=5., type=float, help='clipping threshold for gradient embedding')
parser.add_argument('--clip1', default=2., type=float, help='clipping threshold for residual gradients')
parser.add_argument('--power_iter', default=1, type=int, help='number of power iterations')
parser.add_argument('--num_groups', default=3, type=int, help='number of parameters groups')
parser.add_argument('--num_bases', default=1000, type=int, help='dimension of anchor subspace')

parser.add_argument('--real_labels', action='store_true', help='use real labels for auxiliary dataset')
parser.add_argument('--aux_dataset', default='imagenet', type=str, help='name of the public dataset, [cifar10, cifar100, imagenet]')
parser.add_argument('--aux_data_size', default=2000, type=int, help='size of the auxiliary dataset')


args = parser.parse_args()

assert args.dataset in ['cifar10', 'svhn']
assert args.aux_dataset in ['cifar10', 'cifar100', 'imagenet']
if(args.real_labels):
    assert args.aux_dataset == 'cifar10'

use_cuda = True
best_acc = 0  
start_epoch = 0  
batch_size = args.batchsize

if(args.seed != -1): 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')
## preparing data for training && testing
if(args.dataset == 'svhn'):  ## For SVHN, we concatenate training samples and extra samples to build the training set.
    trainloader, extraloader, testloader, n_training, n_test = get_data_loader('svhn', batchsize = args.batchsize)
    for train_samples, train_labels in trainloader:
        break
    for extra_samples, extra_labels in extraloader:
        break
    train_samples = torch.cat([train_samples, extra_samples], dim=0)
    train_labels = torch.cat([train_labels, extra_labels], dim=0)

else:
    trainloader, testloader, n_training, n_test = get_data_loader('cifar10', batchsize = args.batchsize)
    train_samples, train_labels = None, None
## preparing auxiliary data
num_public_examples = args.aux_data_size
if('cifar' in args.aux_dataset):
    if(args.aux_dataset == 'cifar100'):
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    public_data_loader = torch.utils.data.DataLoader(testset, batch_size=num_public_examples, shuffle=False, num_workers=2) #
    for public_inputs, public_targets in public_data_loader:
        break
else:
    public_inputs = torch.load('imagenet_examples_2000')[:num_public_examples]
if(not args.real_labels):
    public_targets = torch.randint(high=10, size=(num_public_examples,))
public_inputs, public_targets = public_inputs.cuda(), public_targets.cuda()
print('# of training examples: ', n_training, '# of testing examples: ', n_test, '# of auxiliary examples: ', num_public_examples)


print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
sampling_prob=args.batchsize/n_training
steps = int(args.n_epoch/sampling_prob)
sigma, eps = get_sigma(sampling_prob, steps, args.eps, args.delta, rgp=args.rgp)
noise_multiplier0 = noise_multiplier1 = sigma
print('noise scale for gradient embedding: ', noise_multiplier0, 'noise scale for residual gradient: ', noise_multiplier1, '\n rgp enabled: ', args.rgp, 'privacy guarantee: ', eps)

print('\n==> Creating GEP class instance')
gep = GEP(args.num_bases, args.batchsize, args.clip0, args.clip1, args.power_iter).cuda()
## attach auxiliary data to GEP instance
gep.public_inputs = public_inputs
gep.public_targets = public_targets

print('\n==> Creating ResNet20 model instance')
if(args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess  + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        net = resnet20()
        net.cuda()
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
        approx_error = checkpoint['approx_error']
    except:
        print('resume from checkpoint failed')
else:
    net = resnet20() 
    net.cuda()

net = extend(net)

net.gep = gep


num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params/(10**6), 'M')

if(args.private):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

loss_func = extend(loss_func)

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())

def group_params(num_p, groups):
    assert groups >= 1

    p_per_group = num_p//groups
    num_param_list = [p_per_group] * (groups-1)
    num_param_list = num_param_list + [num_p-sum(num_param_list)]
    return num_param_list

print('\n==> Dividing parameters in to %d groups'%args.num_groups)
gep.num_param_list = group_params(num_params, args.num_groups)

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()
    steps = n_training//args.batchsize

    if(train_samples == None): # using pytorch data loader for CIFAR10
        loader = iter(trainloader)
    else: # manually sample minibatchs for SVHN
        sample_idxes = np.arange(n_training)
        np.random.shuffle(sample_idxes)

    for batch_idx in range(steps):
        if(args.dataset=='svhn'):
            current_batch_idxes = sample_idxes[batch_idx*args.batchsize : (batch_idx+1)*args.batchsize]
            inputs, targets = train_samples[current_batch_idxes], train_labels[current_batch_idxes]
        else:
            inputs, targets = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if(args.private):
            logging = batch_idx % 20 == 0
            ## compute anchor subspace
            optimizer.zero_grad()
            net.gep.get_anchor_space(net, loss_func=loss_func, logging=logging)
            ## collect batch gradients
            batch_grad_list = []
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            with backpack(BatchGrad()):
                loss.backward()
            for p in net.parameters():
                batch_grad_list.append(p.grad_batch.reshape(p.grad_batch.shape[0], -1))
                del p.grad_batch
            ## compute gradient embeddings and residual gradients
            clipped_theta, residual_grad, target_grad = net.gep(flatten_tensor(batch_grad_list), logging = logging)
            ## add noise to guarantee differential privacy
            theta_noise = torch.normal(0, noise_multiplier0*args.clip0/args.batchsize, size=clipped_theta.shape, device=clipped_theta.device)
            grad_noise = torch.normal(0, noise_multiplier1*args.clip1/args.batchsize, size=residual_grad.shape, device=residual_grad.device)
            clipped_theta += theta_noise
            residual_grad += grad_noise
            ## update with Biased-GEP or GEP
            if(args.rgp):
                noisy_grad = gep.get_approx_grad(clipped_theta) + residual_grad
            else:
                noisy_grad = gep.get_approx_grad(clipped_theta)
            if(logging):
                print('target grad norm: %.2f, noisy approximation norm: %.2f'%(target_grad.norm().item(), noisy_grad.norm().item()))
            ## make use of noisy gradients
            offset = 0
            for p in net.parameters():
                shape = p.grad.shape
                numel = p.grad.numel()
                p.grad.data = noisy_grad[offset:offset+numel].view(shape) #+ 0.1*torch.mean(pub_grad, dim=0).view(shape)
                offset+=numel
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            try:
                for p in net.parameters():
                    del p.grad_batch
            except:
                pass
        optimizer.step()
        step_loss = loss.item()
        if(args.private):
            step_loss /= inputs.shape[0]
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
    return (train_loss/batch_idx, acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        ## Save checkpoint.
        if acc > best_acc:
            best_acc = acc
            checkpoint(net, acc, epoch, args.sess)

    return (test_loss/batch_idx, acc)


print('\n==> Strat training')

for epoch in range(start_epoch, args.n_epoch):
    lr = adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.n_epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)

try:
    os.mkdir('approx_errors')
except:
    pass
import pickle
bfile=open('approx_errors/'+args.sess+'.pickle', 'wb')
pickle.dump(net.gep.approx_error, bfile)
bfile.close()