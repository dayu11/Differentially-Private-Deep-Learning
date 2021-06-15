import numpy as np

from rdp_accountant import compute_rdp, get_privacy_spent


def loop_for_sigma(q, T, eps, delta, cur_sigma, interval, rdp_orders=32):
    assert eps > 0.
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
## rgp: use residual gradient perturbation or not
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