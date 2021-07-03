import torch
import torch.nn as nn

import numpy as np

def process_batch_grad(batch_grad, scale):
    dim = len(batch_grad.shape)
    scale = scale.view([batch_grad.shape[0]] + [1]*(dim - 1))
    batch_grad.mul_(scale)
    batch_g = torch.sum(batch_grad, dim=0)
    return batch_g 


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
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col

def weight_decomposition(W, rank, iter=1):
    with torch.no_grad():
        outdim, indim = W.shape

        R = torch.normal(0, 1, size = (indim, rank)).cuda().half()
        for _ in range(iter):
            L = torch.matmul(W , R) # outdim x rank
            orthogonalize(L)
            R = torch.matmul(W.T, L) # indim x rank
            orthogonalize(R)

        R = R.T
        residual = W - torch.matmul(L, R)
        return L, R, residual

def normalize_batch_g(batch_g):
    n = batch_g.shape[0]
    flat_batch_g = batch_g.view(n, -1)
    norms = torch.norm(flat_batch_g, dim=1)
    return batch_g / norms.view(n, 1, 1)


def linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0].detach()

def linear_backward_hook(module, grad_input, grad_output):

    grad_output = grad_output[0].detach() # len, n, outdim
    grad_input = module.input #len, n, indim


    if(len(grad_output.shape)==3): # normal layers
        grad_output = grad_output.permute(1, 2, 0) # n, outdim, len
        grad_input = grad_input.permute(1, 0, 2) # n, len, indim

        module.weight.batch_grad = torch.bmm(grad_output, grad_input)
        

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = torch.sum(grad_output, dim=2)

    elif(len(grad_output.shape)==2): #final classification layer
        grad_output = grad_output.view(grad_output.shape[0], grad_output.shape[1], 1) # n, outdim, 1
        grad_input = grad_input.view(grad_input.shape[0], 1, grad_input.shape[1]) # n, 1, indim

        # Recall thatwe do not use reparametrization for the linear classification layer.
        # The gradients of the classification layer are significantly larger than the gradients of low-rank carriers. 
        # So we manually scale down the gradients. The scaling is a constant and is the same for all downstream tasks
        module.weight.batch_grad = torch.bmm(grad_output, grad_input) / 10. 

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = grad_output.view(grad_output.shape[0], grad_output.shape[1]) / 10. 

    else:
        raise 'not implemented error'
class LrkLinear(nn.Module):
    
    def __init__(self, indim, outdim, batch_dim=0):
        super(LrkLinear, self).__init__()

        #self.rank = rank
        self.batch_dim = batch_dim

        tensor = torch.ones(())
        self.weight = nn.Parameter(tensor.new_empty(size=(outdim, indim), dtype=torch.half))

        self.register_forward_hook(linear_forward_hook)
        self.register_backward_hook(linear_backward_hook)    

    def forward(self, x):
        acti = torch.matmul(x, self.weight.T)
        return acti
