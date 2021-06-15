import torch
import torch.nn as nn
import numpy as np
import math
import time

from .utils import _compute_conv_grad_sample, _compute_linear_grad_sample


def conv3x3(in_planes, out_planes, stride=1, group=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, bias=False)


def lrk_conv3x3(in_planes, out_planes, stride=1, rank=1, group=1, kernel_size=3, padding=1):
    return LrkConv2d(in_planes, out_planes, stride=stride, rank=rank, group=group, kernel_size=kernel_size, padding=padding)


def conv1x1(in_planes, out_planes, stride=1, group=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                    bias=False)


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


def weight_decomposition(W, R=None, rank=1):

    outdim, indim = W.shape[0], W.shape[1]

    R = torch.normal(0, 1, size = (indim, rank)).cuda()
    for _ in range(1):
        L = torch.matmul(W , R) # outdim x rank
        orthogonalize(L)
        R = torch.matmul(W.T, L) # indim x rank
        orthogonalize(R)  
    R = R.T
    approx_error = W - torch.matmul(L, R)
    return L, R, approx_error

def ConvFowardHook(module, intsr, outtsr):
    module.input = intsr[0].detach()

def ConvBackwardHook(module, grad_input, grad_output, loss_reduction='mean'):
    gpu_id = torch.cuda.current_device() 
    grad_output = grad_output[0].detach()
    if(loss_reduction == 'mean'):
        n = module.input.shape[0]
        _compute_conv_grad_sample(module, module.input, n*grad_output, gpu_id=gpu_id)
    else:
        _compute_conv_grad_sample(module, module.input, grad_output, gpu_id=gpu_id)




class LrkConv2d(nn.Module):
    
    def __init__(self, inplanes, planes, stride, rank=1, group=1, kernel_size=3, padding=1):
        super(LrkConv2d, self).__init__()

        self.rank = rank

        self.full_conv = conv3x3(inplanes, planes, stride, kernel_size=kernel_size, padding=padding)
        self.full_conv.weight.requires_grad = False

        self.left_layer = conv1x1(self.rank, planes, stride=1, kernel_size=kernel_size, padding=padding)
        self.right_layer = conv3x3(inplanes, self.rank, stride, kernel_size=kernel_size, padding=padding)
        # We find that only using the gradients of the left gradient carrier further reduces the number of trainable parameters and improves accuracy.
        # For example, we achieve ~69% test accuracy on CIFAR10, which is 5% higher than the original result.
        self.right_layer.weight.requires_grad = False

        self.regisiter_hooks(self.left_layer, ConvFowardHook, ConvBackwardHook)
        self.regisiter_hooks(self.right_layer, ConvFowardHook, ConvBackwardHook)

        self.init_weight = [None, self.full_conv.weight.data.clone().cuda()]

        self.approx_error = None
    
    def regisiter_hooks(self, layer, forward_hook, backward_hook):
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    # replace the parameters
    def replace_weight_LR(self, wL, wR):
        self.left_layer.weight.data = wL
        self.right_layer.weight.data = wR
    def replace_weight_full(self, wF):
        self.full_conv.weight.data = wF

    # reconstruct an update from the gradients of low-rank gradient carriers
    def get_full_grad(self, lr):
        # We find that only using the gradients of the left gradient carrier further reduces the number of trainable parameters and improves accuracy.
        # For example, we achieve ~69% test accuracy on CIFAR10, which is 5% higher than the original result.
        grad_shape = self.full_conv.weight.shape
        left_grad = self.left_layer.weight.grad.view(-1, self.rank)
        right_data = self.right_layer.weight.data.view(self.rank, -1)

        left_g_right_w = torch.matmul(left_grad, right_data)

        grad = left_g_right_w 
        grad = grad.view(grad_shape)
        return grad

    # restore the original weight
    def _update_weight(self):
        self.full_conv.weight.data = self.cached_weight

    # update initial weight, used for calculating historical update
    def _update_init_weight(self):
        self.init_weight[0] = self.init_weight[1]/self.init_weight[1].norm()
        self.init_weight[1] = self.full_conv.weight.data.clone()
         
    # run weight decomposition
    def _decomposite_weight(self):
        full_weight = self.full_conv.weight.data

        self.cached_weight = full_weight

        if(self.init_weight[0] != None):
            target = full_weight/full_weight.norm() - self.init_weight[0]
        else:
            target = full_weight

        original_shape = target.shape 
        target = target.view(target.shape[0], -1)

        wL, wR, self.approx_error = weight_decomposition(target, rank=self.rank)
        residual_weight = full_weight.view(full_weight.shape[0], -1) - torch.matmul(wL, wR)

        wR = wR.view(self.rank, *original_shape[1:])
        wL = wL.view(original_shape[0], self.rank, 1, 1)
        residual_weight = residual_weight.view(original_shape)        


        self.residual_weight = residual_weight
        self.replace_weight_LR(wL, wR)
        self.replace_weight_full(residual_weight)

    def forward(self, x):
        if(self.is_training):
            lrk_x = self.right_layer(x)
            lrk_x = self.left_layer(lrk_x)

            residual_x = self.full_conv(x)

            return lrk_x + residual_x
        else:
            return self.full_conv(x)
    def normal_forward(self, x):
        return self.full_conv(x)


use_bn = False


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, rank, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.rank = rank
        self.conv1 = lrk_conv3x3(inplanes, planes, stride, rank=self.rank)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = lrk_conv3x3(planes, planes, rank=self.rank)
        

        if(use_bn):
            self.bn1 = nn.BatchNorm2d(inplanes, affine=False)
            self.bn2 = nn.BatchNorm2d(planes, affine=False)
        else:
            self.bn1 = nn.GroupNorm(min(32, inplanes), inplanes, affine=False)
            self.bn2 = nn.GroupNorm(min(32, planes), planes, affine=False)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        

        if self.downsample is not None:
            identity = self.downsample(x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)
        out += identity
        

        return out


def LinearFowardHook(module, intsr, outtsr):
    module.input = intsr[0].detach()

def LinearBackwardHook(module, grad_input, grad_output, loss_reduction='mean'):
    num_gpu = torch.cuda.device_count()
    if(num_gpu > 1):
        gpu_id = torch.cuda.current_device() 
    else:
        gpu_id = None
    grad_output = grad_output[0].detach()
    if(loss_reduction == 'mean'):
        n = module.input.shape[0]
        _compute_linear_grad_sample(module, module.input, n*grad_output, gpu_id=gpu_id)
    else:
        _compute_linear_grad_sample(module, module.input, grad_output, gpu_id=gpu_id)
    #we do not use reparameterization for the classification layer because it is already low-rank
    #scale down the grad of the final classification layer so that it has approximatly the same norm as low-rank gradients
    module.weight.grad_sample /= 100


class ResNet(nn.Module):

    def __init__(self, block, layers, hyper_params_dict, num_classes=10):
        super(ResNet, self).__init__()
    
        self.rank = hyper_params_dict['rank']
        self.width = hyper_params_dict['width']
        k = self.width

        

        self.num_layers = sum(layers)
        self.inplanes = 16 * k
        self.conv1 = lrk_conv3x3(3, 16 * k, rank=self.rank)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16 * k, layers[0], self.rank)
        self.layer2 = self._make_layer(block, 32 * k, layers[1], self.rank, stride=2)
        self.layer3 = self._make_layer(block, 64 * k, layers[2], self.rank, stride=2)

        if(use_bn):
            self.bn1 = nn.BatchNorm2d(64 * k)
        else:
            self.bn1 = nn.GroupNorm(32, 64 * k, affine=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # we do not reparametrize linear layer because it is already low-rank
        self.fc = nn.Linear(64 * k, num_classes, bias=False)
        self.fc.register_forward_hook(LinearFowardHook)
        self.fc.register_backward_hook(LinearBackwardHook)
        # standard initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.clipping = None
        self.sigma = None

        

    def _make_layer(self, block, planes, blocks, rank, stride=1):
        downsample = None
        if(use_bn):
            norm_layer = nn.BatchNorm2d(self.inplanes)
        else:
            norm_layer = nn.GroupNorm(32, self.inplanes, affine=False)
        if stride != 1:
            downsample = nn.Sequential(
                nn.AvgPool2d(1, stride=stride),
                norm_layer,
            )

        layers = []
        layers.append(block(self.inplanes, planes, rank, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, rank))

        return nn.Sequential(*layers)

    def update_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_weight')):
                m._update_weight()

    def update_init_weight(self):
        for m in self.modules():
            if(hasattr(m, '_update_init_weight')):
                m._update_init_weight()

    def decomposite_weight(self):
        for m in self.modules():
            if(hasattr(m, '_decomposite_weight')):
                m._decomposite_weight()

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def resnet28(num_classes, hyper_params_dict):
    """Constructs a ResNet-20 model.

    """
    model = ResNet(BasicBlock, [4, 4, 4], hyper_params_dict, num_classes=num_classes)
    return model

