import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent


def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32):
    while True:
        orders = np.arange(2, rdp_orders, 0.1)
        steps = T

        rdp = compute_rdp(q, cur_sigma, steps, orders) 
        cur_eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
        if(cur_eps<eps and cur_sigma>interval):
            cur_sigma -= interval
            previous_eps = cur_eps
        else:
            cur_sigma += interval
            break    
    return cur_sigma, previous_eps, opt_order





## interval: init search inerval
def get_sigma(q, T, eps, delta, init_sigma=10, interval=0.5):
    cur_sigma = init_sigma
    
    cur_sigma, _, opt_order = loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
    interval /= 10
    cur_sigma, _, opt_order = loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
    interval /= 10
    cur_sigma, _, opt_order = loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
    interval /= 10
    cur_sigma, previous_eps, opt_order = loop_for_sigma(q, T, eps, delta, cur_sigma, interval)
    return cur_sigma, previous_eps, opt_order




rdp_orders = 4096

def compute_rdp_gd(sensitivity, sigma, orders=range(2, 2+rdp_orders)):

    orders=np.array(orders)
    return (sensitivity**2)*orders/(2*sigma**2)

def rdp_to_eps(rdps, delta, orders=range(2, 2+rdp_orders)):
    orders=np.array(orders)
    eps=np.log(1/delta)/(orders-1)
    eps=rdps+eps
    return eps

def get_sigma_gd(sensitivity, steps, eps, delta):
    sigma=0.1
    while True:
        if(np.min(rdp_to_eps(compute_rdp_gd(sensitivity, sigma)*steps, delta))>eps):
            sigma+=0.1
        else:
            break
    for precision in range(2, 6):
        for i in range(10):
            sigma-=10**(-precision)
            if(np.min(rdp_to_eps(compute_rdp_gd(sensitivity, sigma)*steps, delta))>eps):
                sigma+=10**(-precision)
                break
    return sigma


#print(get_sigma_gd(1, 200, 8, 1e-5))
#exit()
