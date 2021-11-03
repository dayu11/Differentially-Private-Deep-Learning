import numpy as np

from .rdp_accountant import compute_rdp, get_privacy_spent
from prv_accountant import Accountant


def get_eps(q, steps, delta, sigma, mode='moments', rdp_orders=32):
    if(mode == 'moments'):
        orders = np.arange(2, rdp_orders, 0.1)
        rdp = compute_rdp(q, sigma, steps, orders) 
        eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    elif(mode == 'prv'):
        accountant = Accountant(
            noise_multiplier=sigma,
            sampling_probability=q,
            delta=delta,
            eps_error=0.1,
            max_compositions=steps)       
        eps_low, eps_estimate, eps_upper = accountant.compute_epsilon(num_compositions=steps)
        eps = eps_upper
    else:
        print('accounting mode not supportted')
        exit()
    return eps

def loop_for_sigma(q, steps, eps, delta, cur_sigma, interval, mode='moments', rdp_orders=32):
    while True:
        cur_eps = get_eps(q, steps, delta, cur_sigma, mode, rdp_orders)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps


## interval: init search inerval
def get_sigma(q, T, eps, delta, init_sigma=10, interval=0.5, mode='moments'):
    cur_sigma = init_sigma
    
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, mode=mode)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, mode=mode)
    interval /= 10
    cur_sigma, _ = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, mode=mode)
    interval /= 10
    cur_sigma, eps = loop_for_sigma(q, T, eps, delta, cur_sigma, interval, mode=mode)
    return cur_sigma, eps


