from variable_list import V
import API_multi as api
import pruning_API as prune_api
import numpy as np
import time
import os
import csv
#(manual i/p) p_start (in percentage), p_end, p_gap, accstd_flag, std_iter, retrain_time_flag, prune_method
p_init = 40 ###for whole desired range start point (only use for ct)
p_start = 40
p_end = 64
p_gap = 12
## first two function call (when s = 0) just one time **(do not use multi gpu for these two functions)
s = 1
### starting layer for retraining (resume capability)
sl = 0
r_flag = 0 ## 0 means not 'retraining mode'
##maximum retrain count for each layer (meaninful when r_flag = 1)(r_count =2 imagenet, 4 others)
r_count = 0 #4
layer_lr_init = 0.008 ##base learning rate for each layer retraining (0.01 by default)
#(0.05,0.1) combo tinyimagenet for prune 0.8 , (0.04, 0.08 for 0.85 likewise)
###Parameters for fine_tune
##ImageNet
# f_epochs = 120
# f_lr = [0.0008]
# lr_change_freq = 4
# lr_divide_factor = 1.2
##Other Datasets like CIFAR10 fepochs =80, lr_change_freq = 10
f_epochs = 80 #40 #80
f_lr_init =  [0.08] #[0.08]
lr_change_freq = 8 #10
lr_divide_factor_init = 1.8 #1.8
# Starting Epoch in fine tune (s_e_f = 0 by default, otherwise multiple of lr_change_freq)
s_e_f = 0
### Flops counter parameters (input image dimension)

ct = np.int64((p_start - p_init) / p_gap)

##For ImageNet it is 1000, for other cases total image (no. of batches to calculate mean)
n_batch_mean = np.int64(np.ceil(V.dataset.num_train_images/V.b_size))

## Only network is assigned in main function, rest all are assigned in variable list
# net = api.Models(model=V.model_str, num_layers=V.n_l, ).net()
net = api.Models(model=V.model_str, num_layers=V.n_l, num_class= V.n_c).net()
net.restore_checkpoint(V.restore_checkpoint_path)
print(net)
# exit()
### retrain & finetune time for each pruning fraction
def init_array(dim):
    arr = np.zeros(np.int64(dim))
    return arr



f_lr = np.zeros_like(f_lr_init)

if s == 0:
    ### run these two codes with CUDA_VISIBLE_DEVICES=0
    prune_api.create_mean_files(net,n_batch_mean)
    prune_api.save_importance(net)
    # prune_api.feature_mag_and_imp_score(net, n_batch_mean)
    os.makedirs(V.base_path_results + '/final_result', exist_ok=True)
    count = (p_end - p_init)/p_gap
    ###retrain and finetune time initialization
    curr_p, r_time_p, t_time_p, th_p, acc_bft_p, acc_aft_p, acc_o= [init_array(count) for i in range(7)]
    np.savetxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt',np.c_[curr_p, r_time_p, t_time_p],delimiter=",", header='pf, retrain_time, retrain_and_finetune_time',fmt='%f')
    np.savetxt(V.base_path_results + '/final_result/accuracy_details.txt',np.c_[curr_p, acc_o, acc_bft_p, acc_aft_p],delimiter=",", header='pf, actual_acc, prune_retrain_acc, finetune_acc',fmt='%f')
    np.savetxt(V.base_path_results + '/final_result/threshold_details.txt', np.c_[curr_p, th_p], delimiter=",", header='pf, threshold', fmt='%f')
else:
    for p_per in range(p_start,p_end,p_gap):

        ## Configuration for Tiny-Imagenet only
        # if p_per == 80:
        #     layer_lr_init = 0.05
        #     f_lr_init = [0.1]
        # if p_per == 85:
        #     layer_lr_init = 0.04
        #     f_lr_init = [0.08]
        # if p_per == 90:
        #     layer_lr_init = 0.03
        #     f_lr_init = [0.06]
        # if p_per == 95:
        #     layer_lr_init = 0.02
        #     f_lr_init = [0.04]

        ct = np.int64((p_per - p_init) / p_gap)
        layer_lr = layer_lr_init*(ct+1)
        # layer_lr = layer_lr_init * (ct + 1)
        #f_lr[0] = f_lr_init[0]*(ct+1)
        f_lr[0] = f_lr_init[0]
        lr_divide_factor = lr_divide_factor_init
        curr_p, r_time_p, t_time_p = np.loadtxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt',skiprows=1, delimiter=",", unpack= True)
        _, th_p = np.loadtxt(V.base_path_results + '/final_result/threshold_details.txt',skiprows=1, delimiter=",", unpack= True)
        _, acc_o, acc_bft_p, acc_aft_p = np.loadtxt(V.base_path_results + '/final_result/accuracy_details.txt',skiprows=1, delimiter=",", unpack= True)
        ##desired pruning fraction (of filters)
        p = p_per/100
        th = prune_api.find_global_threshold(p)
        start_time = time.time()
        ## Prune_Retrain & Finetune creteas net inside function in API
        te_acc_org, te_acc_prune_retrain = prune_api.prune_retrain_block(p,sl,r_flag,r_count,layer_lr,V.image_dim)
        r_end_time = time.time()
        te_acc_pr_fine = prune_api.finetune_retained_model(p,s_e_f,f_epochs,f_lr,lr_change_freq,lr_divide_factor)
        end_time = time.time()
        retrain_time = r_end_time-start_time
        total_time = end_time-start_time

        if np.isscalar(curr_p) == True:
            curr_p = round(p, 4)
            r_time_p = round(retrain_time, 4)
            t_time_p = round(total_time, 4)
            acc_o = round(te_acc_org[0], 4)
            acc_bft_p = round(te_acc_prune_retrain[0], 4)
            acc_aft_p = round(te_acc_pr_fine[0], 4)
            th_p = th
        else:
            curr_p[ct] = round(p,4)
            r_time_p[ct] = round(retrain_time,4)
            t_time_p[ct] = round(total_time,4)
            acc_o[ct] = round(te_acc_org[0],4)
            acc_bft_p[ct] = round(te_acc_prune_retrain[0],4)
            acc_aft_p[ct] = round(te_acc_pr_fine[0],4)
            th_p[ct] = th

        np.savetxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt', np.c_[curr_p, r_time_p, t_time_p], delimiter=",",
                   header='pf, retrain_time, retrain_and_finetune_time', fmt='%f')
        np.savetxt(V.base_path_results + '/final_result/threshold_details.txt', np.c_[curr_p, th_p], delimiter=",",
                   header='pf, threshold', fmt='%f')
        np.savetxt(V.base_path_results + '/final_result/accuracy_details.txt', np.c_[curr_p, acc_o, acc_bft_p, acc_aft_p], delimiter=",",
                   header='pf, actual_acc, prune_retrain_acc, finetune_acc', fmt='%f')




print("current p",curr_p)
print("finetune timep:",r_time_p)
