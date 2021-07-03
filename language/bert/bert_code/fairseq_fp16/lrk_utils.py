import torch
import torch.nn as nn

import numpy as np

def process_batch_grad(batch_grad, scale):
    dim = len(batch_grad.shape)
    scale = scale.view([batch_grad.shape[0]] + [1]*(dim - 1))
    batch_grad.mul_(scale)
    #print(torch.norm(batch_grad.view(batch_grad.shape[0], -1), dim=1).cpu().numpy())
    batch_g = torch.sum(batch_grad, dim=0)
    #noise = torch.normal(0, 0.414, size=batch_grad.shape[1:]).cuda().half()
    #print('batch g norm: ', batch_g.norm().item(), 'noise norm: ', noise.norm().item())
    return batch_g #+ noise
    #return torch.sum(batch_grad, dim=0)  + torch.normal(0, 62.6, size=batch_grad.shape[1:]).cuda().half()


# def add_noise(params, sigma):
#     for p in params:
#         p.grad += torch.normal(0, sigma, size=p.grad.shape).cuda().half()

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

def weight_decomposition(W, rank, iter=1, mode='rs'):
    with torch.no_grad():
        outdim, indim = W.shape

        if(mode == 'grandom'):
            L = torch.normal(0, 1, size=(outdim, rank)).cuda().half()
            R = torch.normal(0, 1, size=(rank, indim)).cuda().half()
            R = R.T
        elif(mode == 'rs'):
            max_rank = min(outdim, indim)
            selected_ranks = np.random.choice(max_rank, rank, replace=False)
            L = W[:, selected_ranks]
            R = W[selected_ranks, :]
            R = R.T
        elif(mode == 'pi'):
            R = torch.normal(0, 1, size = (indim, rank)).cuda().half()
            for _ in range(iter):
                L = torch.matmul(W , R) # outdim x rank
                #orthogonalize(L)
                #L /= torch.norm(L, dim=0).view(1, -1)
                R = torch.matmul(W.T, L) # indim x rank
                #orthogonalize(R)
            #R = R.T

        if(mode != 'pi'):
            if(rank == 1):
                L /= L.norm()
                R /= R.norm()
            else:
                pass
                #orthogonalize(L)
                #orthogonalize(R)
        R = R.T
        residual = W - torch.matmul(L, R)
        #print(L.shape, R.shape, residual.shape, residual.norm().item())
        return L, R, residual

def normalize_batch_g(batch_g):
    n = batch_g.shape[0]
    flat_batch_g = batch_g.view(n, -1)
    norms = torch.norm(flat_batch_g, dim=1)
    return batch_g / norms.view(n, 1, 1)


def linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0].detach()

def linear_backward_hook(module, grad_input, grad_output):
    #num_gpu = torch.cuda.device_count()
    #if(num_gpu > 1):
    #    gpu_id = torch.cuda.current_device() 
    #else:
    #    gpu_id = None
    # if(gpu_id == 0):
    #     print(gpu_id, 'LinearHook')
    #grad_output = grad_output[0].detach()
    #print(module.input.shape, grad_output.shape, module.weight.shape)
    grad_output = grad_output[0].detach() # len, n, outdim
    grad_input = module.input #len, n, indim

    #grad_output = grad_output.permute(1, 2, 0) # n, outdim, len
    #grad_input = grad_input.permute(1, 0, 2) # n, len, indim

    #module.batch_grad = torch.bmm(grad_output, grad_input)

    #print(module.batch_grad.shape)

    #module.weight.batch_grad = torch.bmm(grad_output, grad_input)

    #if(hasattr(module, 'bias')):
    #    module.bias.batch_grad = torch.sum(grad_output, dim=2)

    if(len(grad_output.shape)==3): # normal layers
        grad_output = grad_output.permute(1, 2, 0) # n, outdim, len
        grad_input = grad_input.permute(1, 0, 2) # n, len, indim

        module.weight.batch_grad = torch.bmm(grad_output, grad_input)
        

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = torch.sum(grad_output, dim=2)

    elif(len(grad_output.shape)==2): #final classification layer
        grad_output = grad_output.view(grad_output.shape[0], grad_output.shape[1], 1) # n, outdim, 1
        grad_input = grad_input.view(grad_input.shape[0], 1, grad_input.shape[1]) # n, 1, indim

        module.weight.batch_grad = torch.bmm(grad_output, grad_input) / 100.

        if(hasattr(module, 'bias')):
            module.bias.batch_grad = grad_output.view(grad_output.shape[0], grad_output.shape[1])

    else:
        raise 'not implemented error'

    #module.weight.batch_grad = normalize_batch_g(module.weight.batch_grad)
class LrkLinear(nn.Module):
    
    def __init__(self, indim, outdim, batch_dim=0):
        super(LrkLinear, self).__init__()

        #self.rank = rank
        self.batch_dim = batch_dim

        tensor = torch.ones(())
        self.weight = nn.Parameter(tensor.new_empty(size=(outdim, indim), dtype=torch.half))

        self.register_forward_hook(linear_forward_hook)
        self.register_backward_hook(linear_backward_hook)    
    # def get_full_grad(self, lr, batch_idx=-1):
    #     #rnd = np.random.rand()
    #     left_g_right_w = torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.data)

    #     m1 = left_g_right_w + torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad)
    #     mg = torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.grad)
    #     m2 = torch.matmul(self.left_layer.weight.data, torch.matmul(self.left_layer.weight.data.T, left_g_right_w))

    #     grad = m1 - m2#torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad)#m1 - m2#lr * mg# - m2 #+ torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad)
    #     if(batch_idx == 0):
    #         print('nl_r+l_nr norm: ', m1.norm().item(), 'l_lt_nl_r norm: ', m2.norm().item(), 'nl_nr norm: ', mg.norm().item())    
    #     # if(rnd>0.5):
    #     #     grad = torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.data) #+ torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad) - lr * torch.matmul(self.left_layer.weight.grad, self.right_layer.weight.grad)
    #     # else:
    #     #     grad = torch.matmul(self.left_layer.weight.data, self.right_layer.weight.grad)
    #     return grad


    def forward(self, x):
        #print(x.shape)
        acti = torch.matmul(x, self.weight.T)#torch.matmul(self.weight, x.T).T
        return acti
