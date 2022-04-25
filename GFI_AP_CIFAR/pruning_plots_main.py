from variable_list import V
import API as api
import pruning_API as prune_api
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from PIL import Image

curr_p, r_time_p, t_time_p = np.loadtxt(V.base_path_results + '/final_result/retrain_and_finetune_time_details.txt',
                                        skiprows=1, delimiter=",", unpack=True)
_, flops_p, params_p, flops_per, params_per = np.loadtxt(V.base_path_results +
                                                         '/final_result/flops_and_params_details.txt', skiprows=1, delimiter=",", unpack=True)
_, th_p = np.loadtxt(V.base_path_results + '/final_result/threshold_details.txt',
                     skiprows=1, delimiter=",", unpack=True)
_, acc_o, acc_bft_p, acc_aft_p = np.loadtxt(V.base_path_results + '/final_result/accuracy_details.txt', skiprows=1,
                                            delimiter=",", unpack=True)


print("current probabilities:", curr_p)
print("current thresholds",th_p)

plt.clf()
fig, axs = plt.subplots(nrows=2, ncols=3)
axs[0,0].plot(curr_p,th_p, color='red')
axs[0,0].set(xlabel='Pruning fraction', ylabel='Imp. score threshold')
axs[0,1].plot(curr_p,flops_per, color='red')
axs[0,1].set(xlabel='Pruning fraction', ylabel='Flops retained(%)')
axs[0,2].plot(curr_p,params_per, color='red')
axs[0,2].set(xlabel='Pruning fraction', ylabel='Params retained(%)')
axs[1,0].plot(curr_p,acc_bft_p, color='red')
axs[1,0].set(xlabel='Pruning fraction', ylabel='Acc. after prune & layerwise retrain(%)')
axs[1,1].plot(curr_p,acc_aft_p, color='red')
axs[1,1].set(xlabel='Pruning fraction', ylabel='Acc. after finetune(%)')
# axs[1,2].plot((curr_p,r_time_p), color='red', label='retrain time')
axs[1,2].plot(curr_p,t_time_p, color='blue', label='total time (retrain & finetune)')
axs[1,2].set(xlabel='Pruning fraction', ylabel='Time (sec.)' )
# axs[1,2].legend()
fig.suptitle(V.dataset_string+'_'+V.model_str+str(V.n_l)+'_prune_all_details')
fig.savefig(V.base_path_results + '/Pruning effect with different pruning fraction.pdf')
# plt.savefig('CIFAR10_All_Class_pmf'+'.pdf')#print(type(dataset))
plt.close(fig)
plt.close()
