import torch
import os
import glob
import pandas
import csv
import numpy as np
import sys
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import API_multi as api
from variable_list import V
from flops_API import get_model_complexity_info
### contains seven functions (first two- only one time call)
##1.create_mean_files
##2.save importance

##3. find_global_threshold
##4. prune_model/prune_retrain_block
##5. finetune_retained_model
##6. retrain_time
##7. flops_params_counter

# Write functions only
# No variables! No loops.


def create_mean_files(net,n_b_m):
    # Create importance files per class per layer
    for l in range(net.max_layers()-V.ig_l):
        features, label_hist = net.get_features(V.dataset, num_batches=n_b_m, after_layer=l, layer_type='conv',
                                                return_type='mean', verbose=True)
        print("layer: ", l)
        features = np.array(features[:V.n_c])
        importance = np.zeros(features.shape[2])
        print("Features Shape:", features.shape)
        os.makedirs(V.base_path_results + "/mean_values/l" + str(l), exist_ok=True)

        for i in range(len(features)):
            importance = np.mean(np.mean(np.mean(np.abs(features[i]), axis=0), axis=1), axis=1)
            np.savetxt(V.base_path_results + '/mean_values/l' + str(l)
                       + '/mean_layer' + str(l) + "class" + str(i) + '.csv', importance, fmt='%f')

def save_importance(net):
    # Create importance files per layer
    C_mean = []

    def read_file(file):
        contents = open(file).read().strip().split('\n')
        contents = [float(x) for x in contents]
        return np.float64(contents)

    root = V.base_path_results + "/mean_values/l0/"

    all_contents = []
    for file_num in range(V.n_c):
        x = read_file(root + 'mean_layer0class' + str(file_num) + '.csv')
        all_contents.append(x)
    all_contents = np.float64(all_contents).T

    for l in range(net.max_layers()-V.ig_l):
        root = V.base_path_results + "/mean_values/l" + str(l) + "/"
        all_contents = []
        for file_num in range(V.n_c):
            x = read_file(root + 'mean_layer' + str(l) + 'class' + str(file_num) + '.csv')
            all_contents.append(x)
        all_contents = np.float64(all_contents).T
        np.savetxt(root + '/concatinated_l' + str(l) + '.csv', all_contents)

    os.makedirs(V.base_path_results + "/imp_val", exist_ok=True)
    overall_importance = []
    for l in range(net.max_layers()-V.ig_l):
        indir = V.base_path_results + "/mean_values/l" + str(l)
        threshold_value = np.loadtxt(indir + '/concatinated_l' + str(l) + '.csv')
        print("layer loaded" + str(l))
        print(threshold_value.shape)
        a = [np.argmax(threshold_value, axis=1), np.max(threshold_value, axis=1)]
        a = np.array(a)

        print(np.array(a).shape)

        importance_layer = []
        for j in range(net.max_filters(layer=l)):
            print(net.max_filters(layer=l))
            c = int(a[0, j])
            d = a[1, j]

            imp = (d)
            importance_layer.append(imp)

        print("importance of imp of layer " + str(l) + "filter", importance_layer)
        print("next layer")
        np.savetxt(V.base_path_results + '/imp_val/layer_' + str(l) + '.csv', importance_layer)
        overall_importance.append(importance_layer)

def find_global_threshold(p):
    # Maximum allowable pruning per layer (limits sometime)

    def concatinate(indir,p):
        fileList = glob.glob(indir + "/*.csv")
        dfList = []
        for filename in fileList:
            print(filename)
            df = pandas.read_csv(filename, header=None)
            dfList.append(df)
        concatDF = pandas.concat(dfList, axis=0).to_numpy()

        concatDF1 = np.sort(concatDF, axis=0)

        for k in range(len(concatDF1) - 1):
            if concatDF1[k] <= concatDF1[k + 1]:
                print("true")
            else:
                break
        # T_V = 0.7 means 70% pruning we want, 0.7>0.6 = 70% pruning is greater than 60%
        # Exception case for a layer: (majority(90%)/all) smaller than thresold -> layer wise thrsold is T_V [[[[GOLDEN RULE]]]]
        # T_V=concatDF1[int(0.55*len(concatDF1))]

        ##### Percentange of Pruning Required #####
        T_V = concatDF1[int(p * len(concatDF1))]
        os.makedirs(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%', exist_ok=True)
        os.makedirs(V.base_path_results + '/Thresold', exist_ok=True)
        np.savetxt(V.base_path_results + '/Thresold' + '/p_f' + str(p) + '.csv', T_V)
        return T_V
    threshold = concatinate(V.base_path_results + "/imp_val",p)

    return threshold

def prune_retrain_block(p,sl,r_flag,r_count,layer_lr,image_dim):
    net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
    g_p = p + (1 - p) / 2
    ### By default set acc_diff to large value (if small then no retraining for non_ImageNet)
    acc_diff = 0.5 ##later used
    def my_loss(output, data, labels):
        labels = api.one_hot(labels, num_class=V.n_c)
        prob = output.softmax(1)
        L = torch.mean(torch.sum(torch.mul(labels, - torch.log(prob + 1e-12)), 1))
        return L

    # Layer Retrain count (default 0) (Otherwise starting layer*temp_count)
    # Uncomment net.restore_pruned_state when sl is not 0
    layer_retrain_count = sl*r_count

    print("sl:", sl)
    print("layer_retrain_count:", layer_retrain_count)

    model_string = V.model_str + str(V.n_l)
    print(net.max_layers())
    ### Manually Providing restore checkpoint path ###
    net.restore_checkpoint(V.restore_checkpoint_path)
    macs, params = get_model_complexity_info(net, image_dim, as_strings=True, print_per_layer_stat=False,
                                             verbose=False)
    print(net.evaluate(V.dataset))
    if V.dataset_string != 'ImageNet':
        tr_acc_org = net.evaluate(V.dataset,train_images= True)
    te_acc_org = net.evaluate(V.dataset, train_images=False)

    if sl == net.max_layers():
        net.restore_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%'
                                 + '/retained_model_after_' + str(sl-1) + 'th_layer_prune')
        net.restore_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' +
                               '/layerwise_prune_retain/pruned_checkpoints/' + 'count'
                               + str(layer_retrain_count) + '.ckpt')
        te_acc_pruned = net.evaluate(V.dataset, train_images=False)
        return te_acc_org, te_acc_pruned


    temp_count = 0
    layer_retrain_indicator = 0
    os.makedirs(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%',
                exist_ok=True)
    os.makedirs(
        V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/layerwise_prune_retain/pruned_states',
        exist_ok=True)
    os.makedirs(V.base_path_results + '/Pruning_Desired ' + str(
        p * 100) + '%' + '/layerwise_prune_retain/pruned_checkpoints',
                exist_ok=True)

    header_1 = ['layer_number','count', 'effective_lr','test_acc','top5_test_acc']
    if sl==0:
        with open(V.base_path_results + '/Pruning_Desired ' + str(
            p * 100) + '%' + '/layerwise_prune_retain/prune_retrain_summary_layer_lr_'+str(layer_lr)+'.csv',
                'wt') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(header_1)

    optim = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4, nesterov=True)
    net.attach_optimizer(optim)
    net.attach_loss_fn(my_loss)
    # to find parameters of layer

    total_param_afterprune = 0
    print("pruning fraction", p)
    threshold_value = np.loadtxt(V.base_path_results + '/Thresold' + '/p_f' + str(p) + '.csv', dtype='float32')
    print("threshold value", threshold_value)

    n_initial = net.num_parameters()
    total_params_initial_till_conv = [0]
    i_f_l, r_f_l, p_r_l, a_tr_l_prev, a_te_l_prev, a_tr_l_curr, a_te_l_curr, c_l = ([] for i in range(8))
    indicator = 'N'

    for j in range(sl, net.max_layers()-V.ig_l):
        desired_layer = j
        # print(net)
        if desired_layer == sl and sl != 0:

            net.restore_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%'
                                     + '/retained_model_after_' + str(sl) + 'th_layer_prune')
            a = net.max_filters(desired_layer)
            i_f_l.append(a)
            # a_tr_temp_prev = net.evaluate(dataset,train_images= True)
            a_te_temp_prev = net.evaluate(V.dataset, train_images=False)
            # a_tr_l_prev.append(a_tr_temp_prev[0])
            a_te_l_prev.append(a_te_temp_prev[0])
        else:
            a = net.max_filters(desired_layer)
            i_f_l.append(a)
            # a_tr_temp_prev = net.evaluate(dataset,train_images= True)
            a_te_temp_prev = net.evaluate(V.dataset, train_images=False)
            # a_tr_l_prev.append(a_tr_temp_prev[0])
            a_te_l_prev.append(a_te_temp_prev[0])
            for i in range(net.max_filters(layer=desired_layer)):
                temp_importance = np.loadtxt(V.base_path_results + '/imp_val/layer_' + str(desired_layer) + '.csv',
                                             dtype='float32')
                ordered_importance = np.sort(temp_importance)
                # print("ordered_importance",ordered_importance)
                ordered_indices = np.argsort(temp_importance)

                if ordered_importance[i] > threshold_value:
                    print("j is", j)
                    print("normal case!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    p_indices = ordered_indices[0:i]
                    indicator = 'N'
                    break

            if ordered_importance[net.max_filters(layer=j) - 1] <= threshold_value:
                print("all small then threshold.....................................................................")
                # p_indices=ordered_indices[0:int(0.32*net.max_filters(layer=j))]
                p_indices = ordered_indices[0:int(g_p * net.max_filters(layer=j))]
                indicator = 'A'

            # if len(p_indices)>V.g_p*(net.max_filters(layer=j)):
            #     print("majority small then threshold....................................................................")
            #     #prunng 80% parameter (70% filters pruning) (a2 + a )/2  = 0.2
            #     p_indices=ordered_indices[0:int(V.g_p * net.max_filters(layer=j))]
            #     indicator = 'M'

            sorted_Arr = np.sort(p_indices)
            p_indices = sorted_Arr[::-1]
            print("p_indices", p_indices)
            # before_param = net.num_parameters()

            if r_flag !=0 and desired_layer > 0:
                net.restore_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' +
                                       '/layerwise_prune_retain/pruned_checkpoints/' + 'count'
                                       + str(layer_retrain_count) + '.ckpt')

            for i in range(len(p_indices)):
                net.prune(layer=desired_layer, filter=p_indices[i], verbose=False)
            net.save_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%'
                                  + '/retained_model_after_' + str(desired_layer) + 'th_layer_prune')


        te_acc = net.evaluate(V.dataset, train_images=False)
        te_acc_temp = te_acc
        print("Test Accuracy before layerwise retraining:", te_acc)
        if r_flag !=0:
            #acc_diff >0.04 and
            while (temp_count <= r_count - 1):
                print("Temporary Test Accuracy during layerwise retraining:", te_acc_temp)
                layer_retrain_count += 1
                temp_count += 1
                if layer_retrain_indicator == 1:
                    net.restore_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' +
                                           '/layerwise_prune_retain/pruned_checkpoints/' + 'count'
                                           + str(layer_retrain_count - 1) + '.ckpt')

                optim = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9, weight_decay=1e-4, nesterov=True)
                net.attach_optimizer(optim)
                lear_rate = (te_acc_org[0] - te_acc_temp[0]) * 0.05 * layer_lr
                # lear_rate = (tr_acc_org[0] - tr_acc_temp[0]) * 0.05 * layer_lr
                net.change_optimizer_learning_rate(lear_rate)
                # net.change_optimizer_learning_rate(0.005)
                net.start_training(V.dataset, np.int(np.ceil(V.dataset.num_train_images / V.b_size)), 1)
                te_acc_temp = net.evaluate(V.dataset, train_images=False)
                layer_retrain_indicator = 1
                net.save_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' +
                                    '/layerwise_prune_retain/pruned_checkpoints/' + 'count' + str(
                    layer_retrain_count) + '.ckpt')

                rows = [desired_layer+1,layer_retrain_count, round(lear_rate,6), round(te_acc_temp[0],3),round(te_acc_temp[1],3)]
                with open(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/layerwise_prune_retain/prune_retrain_summary_layer_lr_'
                          + str(layer_lr) + '.csv','a') as results_file:
                    csv_writer = csv.writer(results_file)
                    csv_writer.writerow(rows)
            else:
                layer_retrain_indicator = 0
                temp_count = 0

            # tr_acc = net.evaluate(V.dataset, train_images=True)
            te_acc = net.evaluate(V.dataset, train_images=False)
            # print("Train Accuracy after layerwise retraining:",tr_acc)
            print("Test Accuracy after layerwise retraining:", te_acc)
            print("filter left", net.max_filters(layer=desired_layer))
            #
            print("acc_after_prunng", net.evaluate(V.dataset))
            b = net.max_filters(desired_layer)
            retain_percent = (b * 100) / a
            print("retain_percentage" + str(retain_percent) + "layer" + str(desired_layer))
            r_f_l.append(b)
            p_r_l.append(retain_percent)
            c_l.append(indicator)
            # a_tr_temp_curr= net.evaluate(V.dataset,train_images= True)
            a_te_temp_curr = net.evaluate(V.dataset, train_images=False)
            # a_tr_l_curr.append(a_tr_temp_curr[0])
            a_te_l_curr.append(a_te_temp_curr[0])

    print("accuracy before training after pruning", net.evaluate(V.dataset))
    n_final = net.num_parameters()
    percent_retan = ((n_final * 100) / n_initial)
    print("percent_remain", percent_retan)

    print("retained_params", n_final)
    overall_retained = (n_final * 100) / n_initial
    print("retained percentage final", overall_retained)
    print("net_param_after train", n_initial)

    print('Saving retained state')
    net.save_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/retained_model')
    tr_acc_pruned = net.evaluate(V.dataset, train_images=True)
    te_acc_pruned = net.evaluate(V.dataset, train_images=False)
    macs_pruned, params_pruned = get_model_complexity_info(net, image_dim, as_strings=True,
                                                           print_per_layer_stat=False,
                                                           verbose=False)
    #### A file should contain the information that how did we get the pruned model
    #### & currently what is the pruned model details (model_pruning_details)
    row1a = ['model', 'dataset', 'n_c', 'batch_size', 'p_f', 'limit_p_f']
    row1b = [model_string, V.dataset_string, V.n_c, V.b_size, p, g_p]
    row2a = ['init_params(million)', 'final_params', 'params_pruned', 'percentage_pruned(%)']
    row2b = [round((n_initial / 1e6), 4), round((n_final / 1e6), 4), round(((n_initial - n_final) / 1e6), 4),
             round((100 - overall_retained), 4)]
    row3a = ['Number of Filters in each layer for Initial Model:']
    row3b = i_f_l
    row4a = ['Number of Filters in each layer for Pruned Model:']
    row4b = r_f_l
    row5a = ['Percentage(%) of Filters in each layer after pruning:']
    row5b = list(np.around(np.array(p_r_l), 3))
    row6a = ['Final Pruning logic applied layerwise :']
    row6b = c_l
    # row7a = ['Training Accuracy of model before pruning current layer:']
    # row7b = list(np.around(np.array(a_tr_l_prev),3))
    # row8a = ['Training Accuracy of model after pruning current layer:']
    # row8b = list(np.around(np.array(a_tr_l_curr),3))
    row9a = ['Testing Accuracy of model before pruning current layer:']
    row9b = list(np.around(np.array(a_te_l_prev), 3))
    row10a = ['Testing Accuracy of model after pruning current layer:']
    row10b = list(np.around(np.array(a_te_l_curr), 3))

    with open(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/model_pruning_details.csv',
              'a') as results_file:
        csv_writer = csv.writer(results_file)
        if r_flag!=0:
            csv_writer.writerow(['Results after layerwise retraining'])
        else:
            csv_writer.writerow(['Results without layerwise retraining'])
        csv_writer.writerow(row1a)
        csv_writer.writerow(row1b)
        csv_writer.writerow(row2a)
        csv_writer.writerow(row2b)
        csv_writer.writerow(row3a)
        csv_writer.writerow(row3b)
        csv_writer.writerow(row4a)
        csv_writer.writerow(row4b)
        csv_writer.writerow(row5a)
        csv_writer.writerow(row5b)
        csv_writer.writerow(row6a)
        csv_writer.writerow(row6b)
        # csv_writer.writerow(row7a)
        # csv_writer.writerow(row7b)
        # csv_writer.writerow(row8a)
        # csv_writer.writerow(row8b)
        csv_writer.writerow(row9a)
        csv_writer.writerow(row9b)
        csv_writer.writerow(row10a)
        csv_writer.writerow(row10b)
        csv_writer.writerow(['No. of Parameters in million for original model: ' + str(n_initial / 1e6)])
        csv_writer.writerow(['No. of Parameters in million for Pruned model: ' + str(n_final / 1e6)])
        # csv_writer.writerow(['Training Accuarcy for original model: ' + str(tr_acc_org)])
        csv_writer.writerow(['Training Accuarcy for pruned model: ' + str(tr_acc_pruned)])
        csv_writer.writerow(['Testing Accuarcy for original model: ' + str(te_acc_org)])
        csv_writer.writerow(['Testing Accuarcy for pruned model: ' + str(te_acc_pruned)])
        csv_writer.writerow(['original flops, pruned flops, original params, pruned params'])
        csv_writer.writerow([macs, macs_pruned, params, params_pruned])
        # csv_writer.writerow(['Pruned model Details: '])
        # csv_writer.writerow([net])
        return te_acc_org, te_acc_pruned
def finetune_retained_model(p,s,epochs,lear1,lear_change_freq,lr_divide_factor):
    best = [0,0]
    net = api.Models(model=V.model_str, num_layers=V.n_l, num_class=V.n_c).net()
    net.restore_checkpoint(V.restore_checkpoint_path)
    print(net.evaluate(V.dataset))
    n_initial = net.num_parameters()
    print("original accuracy", net.evaluate(V.dataset))
    net.restore_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/retained_model')
    # print(net)
    n_final = net.num_parameters()
    percentage_retain = ((n_final * 100) / n_initial)
    print("Total parameters in  million", (n_initial / 1e6))
    print("retained parameters in  million", (n_final / 1e6))

    print("percentage retained parameters", percentage_retain)
    print("accuracy after prune", net.evaluate(V.dataset))

    # Starting Epoch (s = 0 by default, otherwise multiple of lr_change_freq)

    ##In general
    # s = 0
    # epochs = 30
    # lear_change_freq = 5
    # lear1 =[0.0008]


    init_loop_size = int(s / lear_change_freq)
    loop_size = int(epochs / lear_change_freq)
    num_itr_per_epoch = np.int(np.ceil(V.dataset.num_train_images / V.b_size))

    def my_loss(output, data, labels):
        labels = api.one_hot(labels, num_class=V.n_c)
        prob = output.softmax(1)
        L = torch.mean(torch.sum(torch.mul(labels, - torch.log(prob + 1e-12)), 1))
        return L

    os.makedirs(V.base_path_results + '/Pruning_Desired ' + str(
        p * 100) + '%' + '/retrain_hyperparameter_checkpoints/prune_'
                + str(round(100 - percentage_retain, 2)) + 'lr_' + str(lear1[0]) + 'd_f_' + str(
        lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq), exist_ok=True)

    header = ['epoch', 'learning_rate', 'train_acc', 'top5_train_acc', 'test_acc', 'top5_test_acc']

    tr_1_list, te_1_list, tr_5_list, te_5_list, epoch_list = ([] for i in range(5))
    # net = api.Models(model='ResNet', num_layers=50).net()
    # net.restore_pruned_state(V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/retained_model')

    if s == 0:
        with open(
                V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/finetune_accuracy_prune_' + str(
                        round(100 - percentage_retain, 2))
                + 'lr_' + str(lear1[0]) + 'd_f_' + str(lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq) + '.csv',
                'wt') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow(header)
        lear = lear1[0]
    else:
        net.restore_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(
            p * 100) + '%' + '/retrain_hyperparameter_checkpoints/prune_' + str(
            round(100 - percentage_retain, 2)) + 'lr_' + str(lear1[0]) + 'd_f_' + str(
            lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq) + '/epochs_' + str(s) + '.ckpt')
        print("After restoring checkpoint:", net.evaluate(V.dataset))
        lear = lear1[0] / (lr_divide_factor ** init_loop_size)

    for i in range(len(lear1)):

        # lear=lear1[i]
        optim = torch.optim.SGD(net.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4, nesterov=True)
        net.attach_optimizer(optim)
        net.attach_loss_fn(my_loss)
        for l in range(loop_size - init_loop_size):
            ##lear = lear/5
            print("lear", lear)

            net.change_optimizer_learning_rate(lear)
            for j in range(1, lear_change_freq + 1):
                net.start_training(V.dataset, num_itr_per_epoch, 1)
                net.save_checkpoint(V.base_path_results + '/Pruning_Desired ' + str(
                    p * 100) + '%' + '/retrain_hyperparameter_checkpoints/prune_' + str(
                    round(100 - percentage_retain, 2)) + 'lr_' + str(lear1[0]) + 'd_f_' + str(
                    lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq) + '/epochs_' + str(
                    (l * lear_change_freq) + j + s) + '.ckpt')
                tr_acc = net.evaluate(V.dataset, train_images=True)
                te_acc = net.evaluate(V.dataset, train_images=False)
                if best[0]< te_acc[0]:
                    best[0] = te_acc[0]
                    best[1] = te_acc[1]
                tr_1_list.append(tr_acc[0])
                tr_5_list.append(tr_acc[1])
                te_1_list.append(te_acc[0])
                te_5_list.append(te_acc[1])
                print("Epochs_Completed", (l * lear_change_freq) + j + s)
                epoch_list.append((l * lear_change_freq) + j + s)
                row = [(l * lear_change_freq) + j + s, round(lear, 9), round(tr_acc[0], 4), round(tr_acc[1], 4),
                       round(te_acc[0], 4), round(te_acc[1], 4)]
                with open(V.base_path_results + '/Pruning_Desired ' + str(
                        p * 100) + '%' + '/finetune_accuracy_prune_' + str(round(100 - percentage_retain, 2))
                          + 'lr_' + str(lear1[0]) + 'd_f_' + str(lr_divide_factor) + 'l_r_c_f_' + str(
                    lear_change_freq) + '.csv', 'a') as results_file:
                    csv_writer = csv.writer(results_file)
                    csv_writer.writerow(row)
            ##lear = lear /2.2
            lear = lear / lr_divide_factor

    plt.plot(epoch_list, tr_1_list, label='Tr_top1_acc')
    plt.plot(epoch_list, te_1_list, label='Te_top1_acc')
    plt.plot(epoch_list, tr_5_list, label='Tr_top5_acc')
    plt.plot(epoch_list, te_5_list, label='Te_top5_acc')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title('Finetune_Accuracy for prune_' + str(round(100 - percentage_retain, 2)) + 'lr_' + str(
        round(lear1[0], 6)) + 'd_f_' + str(lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq))
    plt.legend()
    plt.savefig(
        V.base_path_results + '/Pruning_Desired ' + str(p * 100) + '%' + '/Finetune_Accuracy for prune_' + str(
            round(100 - percentage_retain, 2)) + 'lr_' + str(round(lear1[0], 6)) + 'd_f_' + str(
            lr_divide_factor) + 'l_r_c_f_' + str(lear_change_freq) + '.pdf')

    return best ##te_acc or best


# net = api.ResNet()
# v = V(net)
# create_mean_files(v)