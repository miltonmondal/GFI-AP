import torch
import os
import csv
import glob
import pandas
import numpy as np
import sys
import torch.optim as optim
import numpy as np
import API as api
from sl_variables import V



def my_loss(output, data, labels):
    labels = api.one_hot(labels, num_class=V.n_c)
    prob = output.softmax(1)
    L = torch.mean(torch.sum(torch.mul(labels, - torch.log(prob+1e-9)), 1))
    return L

def sl_prune(p, trial, epochs, lear_change_freq, lear1, lr_divide_factor, lr_change_freq):
    net = api.Models(model=V.model_str, num_layers=V.n_l, num_transition_shape=1 * 1, num_linear_units=512,
                     num_class=V.n_c).net()
    net.restore_checkpoint(V.restore_checkpoint_path)
    acc = net.evaluate(V.dataset, train_images=True)
    print("Pretrained training accuracy:", acc[0])

    file_str = V.base_path_results+'/Results_After_Pruning_trial_'+str(trial)+'.csv'

    num_itr_per_epoch = np.int(np.ceil(V.dataset.num_train_images / V.b_size))
    n_initial = net.num_parameters()

    header = ['layer', 'Acc_B', 'Acc_A', 'Filters', 'Retain']
    with open(file_str, 'wt') as results_file:
        csv_writer = csv.writer(results_file)
        csv_writer.writerow(header)

    for layer in range(net.max_layers()- V.ig_l):
        j = layer
        no_filters = np.int(p * net.max_filters(layer=j))
        print("No. of filters for pruning:", no_filters)
        for i in range(no_filters):
            net.prune(layer=j, filter=i, verbose=False)
        os.makedirs(V.base_path_results+"/Pruned_States/layer_" + str(j), exist_ok=True)
        print('Saving Pruned state')
        net.save_pruned_state(V.base_path_results+"/Pruned_States/layer_" + str(j) + "/importance_retained_state-001")
        acc_prune, _ = net.evaluate(V.dataset)
        filter_left = net.max_filters(layer=j)
        print("filter left", filter_left)
        print("acc_after_pruning", acc_prune)
        n_final = net.num_parameters()
        percent_retain = ((n_final * 100) / n_initial)
        print("Total Parameters of current model in millions:", (n_final / 1000000))
        print("percent_remain", percent_retain)

        os.makedirs(V.base_path_results+"/Fine_Tune_Checkpoints/layer_" + str(j), exist_ok=True)
        net = api.Models(model=V.model_str, num_layers=V.n_l, num_transition_shape=1 * 1, num_linear_units=512,
                         num_class=V.n_c).net()
        net.restore_pruned_state(V.base_path_results+"/Pruned_States/layer_" + str(j) + "/importance_retained_state-001")
        optim = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)
        net.attach_optimizer(optim)
        net.attach_loss_fn(my_loss)
        for e in range(0, epochs, lr_change_freq):
            if e == 0:
                lear = lear1
            else:
                lear = lear / lr_divide_factor
            net.save_checkpoint(V.base_path_results+"/Fine_Tune_Checkpoints/layer_" + str(j) + "/epoch_" + str(e) + '.ckpt')
            print("learning rate:", lear)
            net.change_optimizer_learning_rate(lear)
            net.start_training(V.dataset, num_itr_per_epoch, lear_change_freq)

        acc_finetune, _ = net.evaluate(V.dataset)
        print("No. of Filters retained in layer " + str(j) + "is: " + str(net.max_filters(layer=j)))
        print("Accuracy after " + str(epochs) + " epochs: " + str(acc_finetune))
        row = [j, round(acc_prune, 2), round(acc_finetune, 2), filter_left, round(percent_retain, 2)]
        with open(file_str, 'a') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(row)
        net = api.Models(model=V.model_str, num_layers=V.n_l, num_transition_shape=1 * 1, num_linear_units=512,
                         num_class=V.n_c).net()
        net.restore_checkpoint(V.restore_checkpoint_path)