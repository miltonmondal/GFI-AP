# import single gpu api
# import flops counter
import API as api ###This single-gpu API is used to count flops
import numpy as np
from variable_list import V
from flops_API import get_model_complexity_info

p_init = 40 ###for whole desired range start point (only use for ct)
p_start = 40
p_end = 64
p_gap = 12
def init_array(dim):
    arr = np.zeros(np.int64(dim))
    return arr


def flops_parameters_count(net,p,image_dim):
    macs, params = get_model_complexity_info(net, image_dim, as_strings=False, print_per_layer_stat=False,
                                             verbose=False)
    net.restore_pruned_state(V.base_path_results +
                             '/Pruning_Desired ' + str(p * 100) +
                             '%' + '/retained_model', arch_only= True)
    macs_pruned, params_pruned = get_model_complexity_info(net, image_dim, as_strings=False,
                                                           print_per_layer_stat=False,
                                                           verbose=False)

    return macs/1e6, macs_pruned/1e6, params/1e6, params_pruned/1e6 #### in million

count = (p_end - p_init)/p_gap

# create network

curr_p, flops_p, flops_per, params_p, params_per= [init_array(count) for i in range(5)]

for p_per in range(p_start, p_end, p_gap):
    net = api.Models(model=V.model_str, num_layers=V.n_l).net()
    p = p_per / 100
    ct = np.int64((p_per - p_init) / p_gap)
    macs, macs_pruned, params, params_pruned = flops_parameters_count(net, p, V.image_dim)
    curr_p[ct] = p
    flops_p[ct] = round(macs_pruned,4)
    params_p[ct] = round(params_pruned,4)
    flops_per[ct] = round((100 * macs_pruned) / macs,4)
    params_per[ct] = round((100 * params_pruned) / params,4)

np.savetxt(V.base_path_results + '/final_result/flops_and_params_details.txt',
           np.c_[curr_p, flops_p, params_p, flops_per, params_per], delimiter=",",
           header='pf, retained_flops(M), retained_params(M), retained_flops(%), retained_params(%)',
           fmt='%.4f')
