import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision, sys, math, copy, os, tqdm, sklearn.metrics, platform, getpass, hashlib, multiprocessing
import matplotlib.pyplot as plt


class Models:

    def __init__(self, model, num_layers, num_transition_shape=None, num_linear_units=None, num_class=None):
        self._model_list = ['VGG', 'ResNet']
        assert model in self._model_list, 'Model must be either ' + ' or '.join(self._model_list)
        self._model = model
        self._num_layers = num_layers
        self._num_transition_shape = num_transition_shape
        self._num_linear_units = num_linear_units
        self._num_class = num_class
        if model == 'ResNet':
            if (num_transition_shape != None) or (num_linear_units != None) or (num_class != None):
                print('Warning: ResNet ignores num_transition_shape, num_linear_units, num_class.')

    def net(self):
        if self._model == 'VGG':
            return self._VGG_(self._num_layers, self._num_transition_shape, self._num_linear_units, self._num_class).to('cuda:0')
        if self._model == 'ResNet':
            return self._ResNet_(self._num_layers).to('cuda:0')

    def generate_graph(self, dataset):
        for data, _ in dataset.train_images:
            torch.onnx.export(net, data.cuda(), 'model_graph.onnx')
            break

    class _Agnostic_():
        def save_state(self):
            print('Saving state')
            torch.save(self.state_dict(), './tmp_ckpt')
            return torch.load('./tmp_ckpt')
        def load_state(self, state):
            print('Loading state')
            self.load_state_dict(state)
        def predict(self, images):
            with torch.no_grad():
                return self.forward(images)
        def start_training(self, dataset, eval_freq, epoch):
            if self._optim == None: raise ConnectionError('Optimizer not attached. Use attach_optimizer to connect optimizer.')
            if self._loss_fn == None: raise ConnectionError('Loss function not attached. Use attach_loss_fn to connect loss_fn')
            collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
            for e in range(1, epoch+1):
                for data, labels in dataset.train_images:
                    data, labels = data.cuda(), labels.cuda()
                    self.train()
                    self._optim.zero_grad()
                    output = self.forward(data)
                    loss = self._loss_fn(output, data, labels)
                    loss.backward()
                    self._optim.step()
                    self._global_step += 1
                    collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('\rIter:%i' % self._global_step)
                    sys.stdout.flush()
                    self._collected_loss.append(loss.item())
                    if self._global_step % eval_freq == 0:
                        training_accuracy = 100*(correct_predictions/total_processed).item()
                        top1, top5 = self.evaluate(dataset)
                        print(') Loss:%.4f  Acc(Train Eval Top5): %.2f %.2f %.2f' % (
                            collate_loss.item()/total_processed,
                            training_accuracy,
                            top1, top5
                        ))
                        collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                        self._collected_metrics.append([training_accuracy, top1, top5])
                    # Make plots
                    self._collected_metrics_array = np.float32(self._collected_metrics).T
                    fig, axs = plt.subplots(2,2)
                    axs[0,0].plot(self._collected_loss), axs[0,0].set_title('Loss')
                    axs[0,1].plot(self._collected_metrics_array[0,:], 'tab:orange'), axs[0,1].set_title('Training Acc')
                    axs[1,0].plot(self._collected_metrics_array[1,:], 'tab:green'), axs[1,0].set_title('Top 1', y=-0.3)
                    axs[1,1].plot(self._collected_metrics_array[2,:], 'tab:red'), axs[1,1].set_title('Top 5', y=-0.3)
                    plt.savefig('./.temp_fig.png')
                    plt.close(fig)
                    os.system('cp ./.temp_fig.png ./plot.png')
                print('\rEpoch', e, 'done.')
            print('Final eval accuracy (Top1,Top5):', self.evaluate(dataset))


        def distill_training(self, dataset, eval_freq, epoch, teacher):
            if self._optim == None: raise ConnectionError('Optimizer not attached. Use attach_optimizer to connect optimizer.')
            if self._loss_fn == None: raise ConnectionError('Loss function not attached. Use attach_loss_fn to connect loss_fn')
            collate_loss, correct_predictions, total_processed = torch.zeros(1).cuda(), torch.zeros(
                1).cuda(), torch.zeros(1).cuda()
            for e in range(1, epoch+1):
                for data, labels in dataset.train_images:
                    data, labels = data.cuda(), labels.cuda()
                    self.train()
                    self._optim.zero_grad()
                    output = self.forward(data)
                    teacher_op = teacher.forward(data)
                    loss = self._loss_fn(output, data, labels, teacher_op)
                    loss.backward()
                    self._optim.step()
                    self._global_step += 1
                    collate_loss += loss
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    total_processed += labels.size()[0]
                    sys.stdout.write('\rIter:%i' % self._global_step)
                    sys.stdout.flush()
                    self._collected_loss.append(loss.item())
                    if self._global_step % eval_freq == 0:
                        training_accuracy = 100*(correct_predictions/total_processed).item()
                        top1, top5 = self.evaluate(dataset)
                        print(') Loss:%.4f  Acc(Train Eval Top5): %.2f %.2f %.2f' % (
                            collate_loss.item()/total_processed,
                            training_accuracy,
                            top1, top5
                        ))
                        collate_loss, correct_predictions, total_processed = torch.zeros(1), torch.zeros(1), torch.zeros(1)
                        self._collected_metrics.append([training_accuracy, top1, top5])
                print('\rEpoch', e, 'done.')
            print('Final eval accuracy (Top1,Top5):', self.evaluate(dataset))

        def attach_loss_fn(self, loss_fn):
            self._loss_fn = loss_fn
        def init_conv_layers(self):
            print('Initializing conv layers')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    class _ResNet_(nn.Module, _Agnostic_):

        class B1(nn.Module):
            def __init__(self, f, dummy_var=False):
                super().__init__()
                self.b1 = nn.Sequential(
                    nn.Conv2d(f, f, 3, 1, 1),
                    nn.BatchNorm2d(f),
                    nn.ReLU(),
                    nn.Conv2d(f, f, 3, 1, 1),
                    nn.BatchNorm2d(f)
                )
            def forward(self, x):
                return F.relu(self.b1(x) + x)

        class B3(nn.Module):
            def __init__(self, f, first_block=False):
                super().__init__()
                if first_block:
                    fin, fout, s = f//2, f, 2
                else:
                    fin, fout, s = f, f, 1
                self.b3 = nn.Sequential(
                    nn.Conv2d(fin, fout, 3, s, 1),
                    nn.BatchNorm2d(fout),
                    nn.ReLU(),
                    nn.Conv2d(fout, fout, 3, 1, 1),
                    nn.BatchNorm2d(fout)
                )
                self.b3_skip = nn.Sequential(
                    nn.Conv2d(fin, fout, 3, s, 1),
                    nn.BatchNorm2d(fout)
                )
            def forward(self, x):
                return F.relu(self.b3(x) + self.b3_skip(x))

        class B4(nn.Module):

            class PaddedAdd(nn.Module):
                def __init__(self, first_block=False):
                    super().__init__()
                    self.first_block = first_block

                def forward(self, x, y):
                    if self.first_block:
                        x = F.max_pool2d(x, 2)

                    if x.shape[1] < y.shape[1]:
                        x = torch.cat((x, torch.zeros(x.shape[0], y.shape[1] - x.shape[1], x.shape[2], x.shape[3], device='cuda:0')), dim=1)
                    elif x.shape[1] > y.shape[1]:
                        y = torch.cat((y, torch.zeros(y.shape[0], x.shape[1] - y.shape[1], y.shape[2], y.shape[3], device='cuda:0')), dim=1)
                    else:
                        pass
                    return x + y

            class Identity(nn.Module):
                def __init__(self, f, first_block=False):
                    super().__init__()
                    self.first_block = first_block
                    self.mat = torch.eye(n=f, device='cuda:0')

                def forward(self, x):
                    if self.first_block:
                        x = F.max_pool2d(x, 2)
                    y = 1

            def __init__(self, f, first_block=False):
                super().__init__()
                if first_block:
                    fin, fout, s = f//2, f, 2
                else:
                    fin, fout, s = f, f, 1
                self.b4 = nn.Sequential(
                    nn.Conv2d(fin, fout, 3, s, 1),
                    nn.BatchNorm2d(fout),
                    nn.ReLU(),
                    nn.Conv2d(fout, fout, 3, 1, 1),
                    nn.BatchNorm2d(fout)
                )
                self.b4_add = self.PaddedAdd(first_block)

            def forward(self, x):
                return F.relu(self.b4_add(x, self.b4(x)))

        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
                self.classifier = nn.Linear(8*8, 10)
            def forward(self, x):
                x = self.gap(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x

        def B_x(self, x, n, f):
            Bx = eval('self.B' + str(x))
            super_block = nn.Sequential(
                Bx(f, True)
            )
            for i in range(n - 1):
                super_block.add_module(str(i + 1), Bx(f))
            return super_block

        def __init__(self, num_layers):
            super().__init__()

            assert num_layers in [18, 34, 50, 101, 152, 20, 32, 44, 56, 110], 'num_layers must be [18, 34, 50, 101, 152, 20, 32, 44, 56, 110]'

            self._global_step = 0
            self._checkpoint = './weights/resnet' + str(num_layers) + '.ckpt'
            self._pruned_state = []
            self._optim = None
            self._idx, self._lin_idx = [], []
            self._tree = [[], []]
            self._collected_loss, self._collected_metrics = [], [[0,0,0]]
            self._loss_fn = None

            # For ImageNet
            if num_layers in [18, 34, 50, 101, 152]:
                self.resnet = eval('torchvision.models.resnet' + str(num_layers) + '()')
                self._num_class, self._num_linear_units = 1000, 512
                self.modules = list(self.modules())
                module_type = [type(m).__name__ for m in self.modules]
                i = 0
                while i < len(self.modules):
                    m_type = module_type[i]
                    if m_type == 'Sequential' and module_type[i+1] == 'Bottleneck':
                        self._idx.append((i+2, 'C'))
                        self._idx.append((i+4, 'C'))
                        if self.modules[i+4].stride == (2,2):
                            self.modules[i+2].stride = (2,2)
                            self.modules[i+4].stride = (1,1)
                        i += 12
                    elif m_type == 'Bottleneck':
                        self._idx.append((i+1, 'C'))
                        self._idx.append((i+3, 'C'))
                        i += 8
                    elif m_type == 'BasicBlock':
                        self._idx.append((i+1, 'C'))
                        i += 6
                    else:
                        i += 1

            # For CIFAR10
            resnet_n = (num_layers - 2) // 6
            if num_layers in [20, 32, 44, 56, 110]:
                self.resnet = nn.Sequential(
                    nn.Conv2d(3, 16, 3, 1, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    self.B_x(x=1, n=resnet_n, f=16),
                    self.B_x(x=4, n=resnet_n, f=32),
                    self.B_x(x=4, n=resnet_n, f=64),
                    self.Classifier()
                )
                self._num_class, self._num_linear_units = 10, 64
                self.modules = list(self.modules())
                module_type = [type(m).__name__ for m in self.modules]
                i = 0
                while i < len(self.modules):
                    m_type = module_type[i]
                    if m_type == 'B1':
                        self._idx.append((i+2, 'C'))
                        i += 7
                    elif m_type == 'B3':
                        self._idx.append((i+2, 'C'))
                        self._idx.append((i+5, 'D'))
                        self._idx.append((i+8, 'S'))
                        i += 10
                    elif m_type == 'B4':
                        self._idx.append((i+2, 'C'))
                        i += 7
                    else:
                        i += 1

            self._super_string = super.__str__(self)
            super_string_split = self._super_string.split('\n')
            for string in super_string_split[1:-2]:
                if string[-1] == '(':
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1].append(0)
                elif string[-2:] == ' )':
                    self._tree[-1].pop(-1)
                    self._tree[-1][-1] += 1
                else:
                    self._tree.append(self._tree[-1].copy())
                    self._tree[-1][-1] += 1

            for i, m in enumerate(self.modules):
                if isinstance(m, nn.Linear):
                    self._lin_idx.append((i, 'L'))

            self.cuda()

        def __str__(self):
            p_str = super().__str__()
            p_str += '\n\n\n----------------------\nIndex wise Module List\n----------------------\n\n'
            p_str += '\n'.join([str(i) + ' ' + type(m).__name__ for i, m in enumerate(self.modules)])
            p_str += '\n\n\n----------------\nPrunable Indices\n----------------\n\n'
            p_str += 'Identifier:ExternalIndex:InternalIndex LayerInfo\n\n'
            for i, (idx, iden) in enumerate(self._idx):
                conv_layer = self.modules[idx]
                p_str += iden + ':' + str(i) + ':' + str(idx) + ' Conv_' + str(conv_layer.kernel_size[0]) \
                         + 'x' + str(conv_layer.kernel_size[1]) + '_s' + str(conv_layer.stride[0]) \
                         + '_(' + str(conv_layer.in_channels) + ', ' + str(conv_layer.out_channels) + ')\n'
            return p_str

        def get_layer_from_tree(self, internal_idx):
            def u(module, k):
                if len(k) == 1:
                    return list(module.named_children())[k[0]][1]
                else:
                    kk = [k[0]]
                    k = k[1:]
                    return u(list(module.named_children())[kk[0]][1], k)
            return u(self.resnet, self._tree[internal_idx])

        def forward(self, x):
            return self.resnet(x)

        def attach_optimizer(self, optim):
            self._optim = optim

        def change_optimizer_learning_rate(self, lr):
            self._optim.param_groups[0]['lr'] = lr

        def restore_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            try:
                self.load_state_dict(torch.load(location))
            except:
                self.resnet.load_state_dict(torch.load(location))
            print('Restoring checkpoint from', location)

        def save_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            torch.save(self.state_dict(), location)
            print('Checkpoint saved at', location)

        def num_parameters(self, layer=None):
            if layer == None:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            else:
                idx, iden = self._idx[layer]
                return sum(p.numel() for p in self.modules[idx].parameters())

        def max_layers(self):
            return len(self._idx)

        def max_filters(self, layer):
            idx, iden = self._idx[layer]
            return self.modules[idx].out_channels

        def dry_run(self, dataset, num_iterations=10):
            if self._optim == None: raise ConnectionError('Optimizer not connected. Use attach_optimizer to connect optimizer.')
            iteration = 0
            for data, labels in dataset.train_images:
                data, labels = data.cuda(), labels.cuda()
                output = self.forward(data)
                loss = nn.CrossEntropyLoss()(output, labels)
                loss.backward()
                self._optim.step()
                sys.stdout.write('\rIter:%i' % iteration)
                sys.stdout.flush()
                if iteration == num_iterations:
                    break
                else:
                    iteration += 1
            print(' All OK!')

        def evaluate(self, dataset, verbose=False, confusion_matrix=False, train_images=False, mode='eval'):
            if mode == 'eval':
                self.eval()
            elif mode == 'train':
                self.train()
            if train_images == False:
                dataset_images = dataset.eval_images
                dataset_num_images = dataset.num_eval_images
            else:
                dataset_images = dataset.train_images
                dataset_num_images = dataset.num_train_images
            correct_predictions = torch.zeros(1).cuda()
            correct_topk_predictions = torch.zeros(1).cuda()
            collected_labels, collected_predictions = [], []
            with torch.no_grad():
                if verbose:
                    data_generator = tqdm.tqdm(dataset_images)
                else:
                    data_generator = dataset_images
                for data, labels in data_generator:
                    data, labels = data.cuda(), labels.cuda()
                    output = self.forward(data)
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    _, predictions_topk = output.topk(5, 1, True, True)
                    predictions_topk = predictions_topk.t()
                    correct_topk_predictions += predictions_topk.eq(labels.view(1, -1).expand_as(predictions_topk)).sum()

                    if confusion_matrix:
                        collected_labels.extend(list(labels.cpu().numpy()))
                        collected_predictions.extend(list(predictions.view_as(labels).cpu().numpy()))

            if confusion_matrix:
                return 100*(correct_predictions/dataset_num_images).item(),\
                       100*(correct_topk_predictions/dataset_num_images).item(),\
                       sklearn.metrics.confusion_matrix(collected_labels, collected_predictions)
            else:
                return 100*(correct_predictions/dataset_num_images).item(),\
                       100*(correct_topk_predictions/dataset_num_images).item()

        def get_features(self, dataset, num_batches, after_layer, layer_type='conv', return_type='tensor', verbose=False):

            assert layer_type in ['conv', 'bn', 'relu'], 'layer_type must be [\'conv\', \'bn\', \'relu\']'
            assert return_type in ['tensor', 'mean'], 'return_type must be either [' + ' '.join(['tensor', 'mean']) + ']'

            class SaveFeatures:
                def __init__(self, layer):
                    self.handle = layer.register_forward_hook(self.hook_fn)
                def hook_fn(self, layer, input, output):
                    self.features = output
                def remove(self):
                    self.handle.remove()

            if layer_type == 'conv':
                required_layer = self.get_layer_from_tree(self._idx[after_layer][0])
            elif layer_type == 'bn' or layer_type == 'relu':
                required_layer = self.get_layer_from_tree(self._idx[after_layer][0]+1)
            save_features = SaveFeatures(required_layer)

            if return_type == 'tensor':
                features_set, labels_set = [], []
            elif return_type == 'mean':
                features_sum, labels_set = [], []
                [features_sum.append(torch.zeros(1)) for _ in range(self._num_class)]
                [labels_set.append(0) for _ in range(self._num_class)]

            for i, (data, labels) in enumerate(dataset.train_images):

                with torch.no_grad(): self.forward(data.cuda())
                features = save_features.features
                if layer_type == 'relu': features = F.relu(features.clone(), inplace=True)

                if return_type == 'tensor':
                    features_set.append(features.cpu().numpy())
                    labels_set.append(labels.cpu().numpy())
                elif return_type == 'mean':
                    for k, label in enumerate(labels):
                        features_sum[label] = features_sum[label] + features[k:k+1,...].cpu()
                        labels_set[label] += 1

                if verbose:
                    sys.stdout.write('\rFetching features: %i/%i' % (i+1, num_batches))
                    sys.stdout.flush()

                if i == num_batches - 1:
                    break

            save_features.remove()
            print('\n')

            if return_type == 'tensor':
                return np.concatenate(features_set), np.concatenate(labels_set)
            elif return_type == 'mean':
                features_sum = [features_sum[i].__mul__(torch.Tensor([1 / labels_set[i]])).cpu().numpy() if labels_set[i] != 0 else np.float32([0]) for i in range(self._num_class)]
                return features_sum, labels_set

        def get_weights(self, layer, filter=None, grad=False):
            required_layer = self.get_layer_from_tree(self._idx[layer][0])

            conv_weight = required_layer.weight.data.clone()
            if grad: conv_weight = required_layer.weight.grad.data.clone()
            try:
                conv_bias = required_layer.bias.data.clone()
                if grad: conv_bias = required_layer.bias.grad.data.clone()
            except:
                conv_bias = torch.zeros(0)
            if filter is not None:
                conv_weight = conv_weight[filter:filter+1]
                conv_weight = conv_weight.cpu().numpy()
                try:
                    conv_bias = conv_bias[filter:filter+1]
                    conv_bias = conv_bias.cpu().numpy()
                except:
                    pass
            return conv_weight, conv_bias

        def get_gradients(self, layer, filter=None):
            return self.get_weights(layer, filter, grad=True)

        def compute_gradients(self, dataset, num_batches=None, verbose=False):

            if verbose:
                data_generator = tqdm.tqdm(dataset.train_images)
            else:
                data_generator = dataset.train_images

            loss_metric = nn.CrossEntropyLoss()
            self.eval()
            self.zero_grad()

            i = 0
            for data, labels in data_generator:
                data, labels = data.cuda(), labels.cuda()
                output = self.forward(data)
                loss = loss_metric(output, labels)
                loss.backward()
                if i == num_batches:
                    break
                i += 1

        def prune(self, layer, filter, verbose=True):

            def prune_conv_layer(internal_idx, filter, in_out):
                conv0 = self.get_layer_from_tree(internal_idx)
                conv0_in_channels = conv0.in_channels
                conv0_out_channels = conv0.out_channels
                conv0_kernel_size = conv0.kernel_size[0]
                conv0_stride = conv0.stride[0]
                conv0_padding = conv0.padding[0]
                conv0_weight = conv0.weight.data.clone()
                try:
                    conv0_bias = conv0.bias.data.clone()
                except:
                    pass

                if in_out == 'out':
                    conv0_target_weight = delete_index(conv0_weight, at_index=filter)
                    try:
                        conv0_target_bias = delete_index(conv0_bias, at_index=filter)
                        conv0.__init__(conv0_in_channels,
                                      conv0_out_channels - 1,
                                      conv0_kernel_size,
                                      conv0_stride,
                                      conv0_padding)
                    except:
                        self.resnet.layer1._modules['0'].conv1
                        conv0.__init__(conv0_in_channels,
                                      conv0_out_channels - 1,
                                      conv0_kernel_size,
                                      conv0_stride,
                                      conv0_padding,
                                      bias=False)
                elif in_out == 'in':
                    conv0_target_weight = delete_index(conv0_weight, at_index=filter, dim=1)
                    try:
                        conv0_target_bias = conv0_bias

                        conv0.__init__(conv0_in_channels - 1,
                                      conv0_out_channels,
                                      conv0_kernel_size,
                                      conv0_stride,
                                      conv0_padding)
                    except:
                        conv0.__init__(conv0_in_channels - 1,
                                      conv0_out_channels,
                                      conv0_kernel_size,
                                      conv0_stride,
                                      conv0_padding,
                                      bias=False)

                conv0.weight.data = conv0_target_weight
                try:
                    conv0.bias.data = conv0_target_bias
                except:
                    pass

            def prune_batchnorm_layer(internal_idx, filter):
                bn = self.get_layer_from_tree(internal_idx)
                bn_num_features = bn.num_features
                bn_weight = bn.weight.data.clone()
                bn_bias = bn.bias.data.clone()
                bn_running_mean = bn.running_mean.data.clone()
                bn_running_var = bn.running_var.data.clone()

                bn_target_num_features = bn_num_features - 1
                bn_target_weight = delete_index(bn_weight, at_index=filter)
                bn_target_bias = delete_index(bn_bias, at_index=filter)
                bn_target_running_mean = delete_index(bn_running_mean, at_index=filter)
                bn_target_running_var = delete_index(bn_running_var, at_index=filter)

                bn.__init__(bn_target_num_features)
                bn.weight.data = bn_target_weight
                bn.bias.data = bn_target_bias
                bn.running_mean.data = bn_target_running_mean
                bn.running_var.data = bn_target_running_var

            def prune_linear_layer(internal_idx, filter, rc):
                ln = self.get_layer_from_tree(internal_idx)
                ln_in_features = ln.in_features
                ln_out_features = ln.out_features
                ln_weight = ln.weight.data.clone()
                ln_bias = ln.bias.data.clone()

                if rc == 'row':
                    ln_target_weight = delete_index(ln_weight, filter)
                    ln_target_bias = delete_index(ln_bias, filter)
                    ln.__init__(ln_in_features, ln_out_features - 1)
                elif rc == 'col':
                    ln_target_weight = delete_index(ln_weight, filter, dim=1)
                    ln_target_bias = ln_bias
                    ln.__init__(ln_in_features - 1, ln_out_features)
                ln.weight.data = ln_target_weight
                ln.bias.data = ln_target_bias

            def delete_index(tensor, at_index, dim=0):
                if dim == 0:
                    return torch.cat((tensor[:at_index,...], tensor[at_index+1:,...]))
                elif dim == 1:
                    return torch.cat((tensor[:,:at_index,...], tensor[:,at_index+1:,...]), dim=dim)

            first_idx, first_iden = self._idx[layer]
            second_idx, second_iden = first_idx+1, 'B'
            if first_iden == 'C':
                third_idx, third_iden = first_idx+2, 'C'
                if not isinstance(self.get_layer_from_tree(third_idx), nn.Conv2d): third_idx += 1
                pruning_config = first_iden + second_iden + third_iden
            elif first_iden == 'D':
                third_idx, third_iden = self._idx[layer+1]
                fourth_idx, fourth_iden = third_idx+1, 'B'
                try:
                    fifth_idx, fifth_iden = self._idx[layer+2]
                    sixth_idx, sixth_iden = self._idx[layer+4]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden + sixth_iden
                except:
                    fifth_idx, fifth_iden = self._lin_idx[0]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden
            elif first_iden == 'S':
                third_idx, third_iden = self._idx[layer-1]
                fourth_idx, fourth_iden = third_idx+1, 'B'
                try:
                    fifth_idx, fifth_iden = self._idx[layer+1]
                    sixth_idx, sixth_iden = self._idx[layer+3]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden + sixth_iden
                except:
                    fifth_idx, fifth_iden = self._lin_idx[0]
                    pruning_config = first_iden + second_iden + third_iden + fourth_iden + fifth_iden

            assert pruning_config in ['CBC', 'DBSBCS', 'DBSBL', 'SBDBCS', 'SBDBL'], 'Error: No suitable pruning config found.'

            if pruning_config == 'CBC':
                if verbose: print('Pruning config:'+pruning_config, '- conv:out,bn:elem,conv:in at location='+str(filter))
                prune_conv_layer(first_idx, filter, 'out')
                prune_batchnorm_layer(second_idx, filter)
                prune_conv_layer(third_idx, filter, 'in')
            elif pruning_config == 'DBSBCS' or pruning_config == 'SBDBCS':
                if verbose: print('Pruning config:'+pruning_config, '- conv:out,bn:elem,conv:out,bn:elem,conv:in,conv:in at location='+str(filter))
                prune_conv_layer(first_idx, filter, 'out')
                prune_batchnorm_layer(second_idx, filter)
                prune_conv_layer(third_idx, filter, 'out')
                prune_batchnorm_layer(fourth_idx, filter)
                prune_conv_layer(fifth_idx, filter, 'in')
                prune_conv_layer(sixth_idx, filter, 'in')
            elif pruning_config == 'DBSBL' or pruning_config == 'SBDBL':
                if verbose: print('Pruning config:'+pruning_config, '- conv:out,bn:elem,conv:out,bn:elem,lin:col at location='+str(filter))
                prune_conv_layer(first_idx, filter, 'out')
                prune_batchnorm_layer(second_idx, filter)
                prune_conv_layer(third_idx, filter, 'out')
                prune_batchnorm_layer(fourth_idx, filter)
                prune_linear_layer(fifth_idx, filter, 'col')

            self._pruned_state.append((layer, filter))

        def save_pruned_state(self, name):
            try:
                os.makedirs(name)
            except FileExistsError:
                print('Warning: A pruned_state with name='+name+' already exists. Overwriting...')

            file = open(name+'/pruned_state.txt', 'w+')
            for state in self._pruned_state:
                layer, filter = state
                file.write(str(layer)+','+str(filter)+'\n')
            file.close()
            torch.save(self.state_dict(), name+'/pruned_weights.ckpt')

        def restore_pruned_state(self, name, arch_only=False):
            file = open(name+'/pruned_state.txt', 'r').read().strip().split('\n')
            self._pruned_state = []
            for data in file:
                layer, filter = data.strip().split(',')
                layer, filter = int(layer), int(filter)
                self._pruned_state.append((layer, filter))
                self.prune(layer, filter, verbose=False)
            if not arch_only:
                self.load_state_dict(torch.load(name+'/pruned_weights.ckpt'))

    class _VGG_(nn.Module, _Agnostic_):

        def __init__(self, num_layers, num_transition_shape, num_linear_units, num_class):
            super().__init__()

            assert num_layers in [11, 13, 16, 19], 'VGG num_layers != [11, 13, 16, 19]'

            self._global_step = 0
            self._checkpoint = './weights/vgg'+str(num_layers)+'-'+str(num_class)+'.ckpt'
            self._pruned_state = []
            self._optim = None
            self._idx = []
            self._prune_linear_units = num_transition_shape
            self._num_class = num_class
            self._collected_loss, self._collected_metrics = [], [[0,0,0]]
            self._loss_fn = None

            self._fe = eval('torchvision.models.vgg' + str(num_layers) + '_bn().features')

            self._c = nn.Sequential(
                nn.Linear(num_transition_shape*512, num_linear_units),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(num_linear_units, num_linear_units),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(num_linear_units, num_class),
            )

            for i, m in enumerate(self._fe.modules()):
                if isinstance(m, nn.Conv2d):
                    self._idx.append((i-1, 'C'))
            for i, m in enumerate(self._c.children()):
                if isinstance(m, nn.Linear):
                    self._idx.append((i + len(self._fe), 'L'))
            self.cuda()

        def __str__(self):
            conv_modules = self.__get_modules__(children='conv')
            linear_modules = self.__get_modules__(children='linear')
            p_str = super.__str__(self) + '\n\nPrunable Indices\n\n'
            for i, (idx, iden) in enumerate(self._idx):
                if iden == 'C':
                    layer = conv_modules[str(idx)]
                    p_str += str(i) + ': Convolution(' + str(layer.in_channels) + ', ' + str(layer.out_channels) + ')\n'
                if iden == 'L':
                    layer = linear_modules[str(idx-len(self._fe))]
                    p_str += str(i) + ': Linear(' + str(layer.in_features) + ', ' + str(layer.out_features) + ')\n'
            return p_str

        def __get_modules__(self, children='conv'):
            if children == 'conv':
                return list(self.children())[0]._modules
            elif children == 'linear':
                return list(self.children())[1]._modules
            else:
                return None

        def forward(self, x, layer=None, layer_type='conv'):
            if layer is not None:
                idx, iden = self._idx[layer]
                if iden == 'C':
                    if layer_type == 'conv':
                        with torch.no_grad(): return self._fe[:idx + 1](x)
                    elif layer_type == 'bn':
                        with torch.no_grad(): return self._fe[:idx + 2](x)
                    elif layer_type == 'relu':
                        with torch.no_grad(): return self._fe[:idx + 3](x)
                elif iden == 'L':
                    with torch.no_grad():
                        x = self._fe(x)
                        x = x.view(x.size(0), -1)
                        return self._c[:(idx - len(self._fe) + 1)](x)
            else:
                x = self._fe(x)
                x = x.view(x.size(0), -1)
                return self._c(x)

        def attach_optimizer(self, optim):
            self._optim = optim

        def change_optimizer_learning_rate(self, lr):
            self._optim.param_groups[0]['lr'] = lr

        def restore_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            self.load_state_dict(torch.load(location))
            print('Checkpoint restored from', location)

        def save_checkpoint(self, location=None):
            if location is None:
                location = self._checkpoint
            torch.save(self.state_dict(), location)
            print('Checkpoint saved at', location)

        def num_parameters(self, layer=None):
            if layer == None:
                return sum(p.numel() for p in self.parameters() if p.requires_grad)
            else:
                idx, iden = self._idx[layer]
                if iden == 'C':
                    return sum(p.numel() for p in self._fe[idx].parameters())
                elif iden == 'L':
                    return sum(p.numel() for p in self._c[idx-len(self._fe)].parameters())

        def max_layers(self):
            return len(self._idx)

        def max_filters(self, layer):
            idx, iden = self._idx[layer]
            if iden == 'C':
                conv_modules = self.__get_modules__(children='conv')
                return conv_modules[str(idx)].out_channels
            elif iden == 'L':
                linear_modules = self.__get_modules__(children='linear')
                return linear_modules[str(idx - len(self._fe))].out_features

        def evaluate(self, dataset, verbose=False, confusion_matrix=False, train_images=False, mode='eval'):
            if mode == 'eval':
                self.eval()
            elif mode == 'train':
                self.train()
            if train_images == False:
                dataset_images = dataset.eval_images
                dataset_num_images = dataset.num_eval_images
            else:
                dataset_images = dataset.train_images
                dataset_num_images = dataset.num_train_images
            correct_predictions = torch.zeros(1)
            correct_topk_predictions = torch.zeros(1)
            collected_labels, collected_predictions = [], []
            with torch.no_grad():
                if verbose:
                    data_generator = tqdm.tqdm(dataset_images)
                else:
                    data_generator = dataset_images
                for data, labels in data_generator:
                    data, labels = data.cuda(), labels.cuda()
                    output = self.forward(data)
                    predictions = output.argmax(dim=1, keepdim=True)
                    correct_predictions += predictions.eq(labels.view_as(predictions)).sum()
                    _, predictions_topk = output.topk(5, 1, True, True)
                    predictions_topk = predictions_topk.t()
                    correct_topk_predictions += predictions_topk.eq(labels.view(1, -1).expand_as(predictions_topk)).sum()

                    if confusion_matrix:
                        collected_labels.extend(list(labels.cpu().numpy()))
                        collected_predictions.extend(list(predictions.view_as(labels).cpu().numpy()))

            if confusion_matrix:
                return 100*(correct_predictions/dataset_num_images).item(),\
                       100*(correct_topk_predictions/dataset_num_images).item(),\
                       sklearn.metrics.confusion_matrix(collected_labels, collected_predictions)
            else:
                return 100*(correct_predictions/dataset_num_images).item(),\
                       100*(correct_topk_predictions/dataset_num_images).item()

        def get_features(self, dataset, num_batches, after_layer, layer_type='conv', return_type='tensor', verbose=False):

            assert layer_type in ['conv', 'bn', 'relu'], 'layer_type must be either [' + ' '.join(['conv', 'bn', 'relu']) + ']'
            assert return_type in ['tensor', 'mean'], 'return_type must be either [' + ' '.join(['tensor', 'mean']) + ']'

            if return_type == 'tensor':
                features_set, labels_set = [], []
            elif return_type == 'mean':
                features_sum, labels_set = [], []
                [features_sum.append(torch.zeros(1)) for _ in range(self._num_class)]
                [labels_set.append(0) for _ in range(self._num_class)]

            for i, (data, labels) in enumerate(dataset.train_images):

                with torch.no_grad(): features = self.forward(data.cuda(), layer=after_layer, layer_type=layer_type)

                if return_type == 'tensor':
                    features_set.append(features.cpu().numpy())
                    labels_set.append(labels.cpu().numpy())
                elif return_type == 'mean':
                    for k, label in enumerate(labels):
                        features_sum[label] = features_sum[label] + features[k:k+1,...].cpu()
                        labels_set[label] += 1

                if verbose:
                    sys.stdout.write('\rFetching features: %i/%i' % (i+1, num_batches))
                    sys.stdout.flush()

                if i == num_batches-1:
                    break

            print('\n')

            if return_type == 'tensor':
                return np.concatenate(features_set), np.concatenate(labels_set)
            elif return_type == 'mean':
                features_sum = [features_sum[i].__mul__(torch.Tensor([1 / labels_set[i]])).cpu().numpy() if labels_set[i] != 0 else np.float32([0]) for i in range(self._num_class)]
                return features_sum, labels_set

        def get_weights(self, layer, filter=None, grad=False):
            idx, iden = self._idx[layer]

            if iden == 'C':
                conv_modules = self.__get_modules__(children='conv')
                conv_weight = conv_modules[str(idx)].weight.data.clone()
                conv_bias = conv_modules[str(idx)].bias.data.clone()

                if grad:
                    conv_weight = conv_modules[str(idx)].weight.grad.data.clone()
                    conv_bias = conv_modules[str(idx)].bias.grad.data.clone()

                if filter is not None:
                    conv_weight = conv_weight[filter:filter+1]
                    conv_bias = conv_bias[filter:filter+1]

                return conv_weight.cpu().numpy(), conv_bias.cpu().numpy()

            elif iden == 'L':
                linear_modules = self.__get_modules__(children='linear')
                linear_weight = linear_modules[str(idx - len(self._fe))].weight.data.clone()
                linear_bias = linear_modules[str(idx - len(self._fe))].bias.data.clone()

                if grad:
                    linear_weight = linear_modules[str(idx - len(self._fe))].weight.grad.data.clone()
                    linear_bias = linear_modules[str(idx - len(self._fe))].bias.grad.data.clone()

                if filter is not None:
                    linear_weight = linear_weight[filter:filter+1,...]
                    linear_bias = linear_bias[filter:filter+1,...]

                return linear_weight.cpu().numpy(), linear_bias.cpu().numpy()

        def get_gradients(self, layer, filter=None):
            return self.get_weights(layer, filter, grad=True)

        def compute_gradients(self, dataset, num_batches=None, verbose=False):

            if verbose:
                data_generator = tqdm.tqdm(dataset.train_images)
            else:
                data_generator = dataset.train_images

            loss_metric = nn.CrossEntropyLoss()
            self.eval()
            self.zero_grad()

            i = 0
            for data, labels in data_generator:
                data, labels = data.cuda(), labels.cuda()
                output = self.forward(data)
                loss = loss_metric(output, labels)
                loss.backward()
                if i == num_batches:
                    break
                i += 1

        def prune(self, layer, filter, verbose=True):

            first_idx, first_iden = self._idx[layer]
            if first_iden == 'C':
                second_idx, second_iden = self._idx[layer][0]+1, 'B'
                third_idx, third_iden = self._idx[layer+1]
                pruning_config = first_iden + second_iden + third_iden
            elif first_iden == 'L':
                second_idx, second_iden = self._idx[layer+1]
                pruning_config = first_iden + second_iden

            def delete_index(tensor, at_index, dim=0):
                if dim == 0:
                    return torch.cat((tensor[:at_index,...], tensor[at_index+1:,...]))
                elif dim == 1:
                    return torch.cat((tensor[:,:at_index,...], tensor[:,at_index+1:,...]), dim=dim)

            def prune_conv_layer(idx, filter, in_out):
                conv_modules = self.__get_modules__(children='conv')
                c0 = str(idx)
                conv0 = conv_modules[c0]
                conv0_in_channels = conv0.in_channels
                conv0_out_channels = conv0.out_channels
                conv0_kernel_size = conv0.kernel_size[0]
                conv0_stride = conv0.stride[0]
                conv0_padding = conv0.padding[0]
                conv0_weight = conv0.weight.data.clone()
                conv0_bias = conv0.bias.data.clone()

                if in_out == 'out':
                    conv0_target_weight = delete_index(conv0_weight, at_index=filter)
                    conv0_target_bias = delete_index(conv0_bias, at_index=filter)

                    conv_modules[c0] = nn.Conv2d(conv0_in_channels,
                                                 conv0_out_channels - 1,
                                                 conv0_kernel_size,
                                                 conv0_stride,
                                                 conv0_padding)
                elif in_out == 'in':
                    conv0_target_weight = delete_index(conv0_weight, at_index=filter, dim=1)
                    conv0_target_bias = conv0_bias

                    conv_modules[c0] = nn.Conv2d(conv0_in_channels - 1,
                                                 conv0_out_channels,
                                                 conv0_kernel_size,
                                                 conv0_stride,
                                                 conv0_padding)

                conv_modules[c0].weight.data = conv0_target_weight
                conv_modules[c0].bias.data = conv0_target_bias

            def prune_batchnorm_layer(idx, filter):
                conv_modules = self.__get_modules__(children='conv')
                b0 = str(idx)
                bn = conv_modules[b0]
                bn_num_features = bn.num_features
                bn_weight = bn.weight.data.clone()
                bn_bias = bn.bias.data.clone()
                bn_running_mean = bn.running_mean.data.clone()
                bn_running_var = bn.running_var.data.clone()

                bn_target_num_features = bn_num_features - 1
                bn_target_weight = delete_index(bn_weight, at_index=filter)
                bn_target_bias = delete_index(bn_bias, at_index=filter)
                bn_target_running_mean = delete_index(bn_running_mean, at_index=filter)
                bn_target_running_var = delete_index(bn_running_var, at_index=filter)

                conv_modules[b0] = nn.BatchNorm2d(bn_target_num_features)
                conv_modules[b0].weight.data = bn_target_weight
                conv_modules[b0].bias.data = bn_target_bias
                conv_modules[b0].running_mean.data = bn_target_running_mean
                conv_modules[b0].running_var.data = bn_target_running_var

            def prune_linear_layer(idx, filter, rc, num_units=1):
                linear_modules = self.__get_modules__(children='linear')
                l0 = str(idx - len(self._fe))
                ln = linear_modules[l0]
                ln_in_features = ln.in_features
                ln_out_features = ln.out_features
                ln_weight = ln.weight.data.clone()
                ln_bias = ln.bias.data.clone()

                if rc == 'row':
                    ln_target_weight = delete_index(ln_weight, filter)
                    ln_target_bias = delete_index(ln_bias, filter)

                    linear_modules[l0] = nn.Linear(ln_in_features, ln_out_features - 1)
                elif rc == 'col':
                    ln_target_weight = ln_weight
                    for _ in range(num_units): ln_target_weight = delete_index(ln_target_weight, filter, dim=1)
                    ln_target_bias = ln_bias

                    linear_modules[l0] = nn.Linear(ln_in_features - num_units, ln_out_features)

                linear_modules[l0].weight.data = ln_target_weight
                linear_modules[l0].bias.data = ln_target_bias

            if pruning_config == 'CBC':
                prune_conv_layer(first_idx, filter, 'out')
                prune_batchnorm_layer(second_idx, filter)
                prune_conv_layer(third_idx, filter, 'in')
                if verbose: print('Pruning conv(out):fil='+str(filter)+', bn:elem='+str(filter)+', conv(in):idx='+str(filter))
            elif pruning_config == 'CBL':
                prune_conv_layer(first_idx, filter, 'out')
                prune_batchnorm_layer(second_idx, filter)
                prune_linear_layer(third_idx, filter, 'col', self._prune_linear_units)
                if verbose: print('Pruning conv(out):fil='+str(filter)+', bn:elem='+str(filter)+', linear(col):idx='+str(filter))
            elif pruning_config == 'LL':
                prune_linear_layer(first_idx, filter, 'row')
                prune_linear_layer(second_idx, filter, 'col')
                if verbose: print('Pruning linear(row):idx='+str(filter)+', bn:elem='+str(filter)+', linear(col):idx='+str(filter))

            self._pruned_state.append((layer, filter))

        def save_pruned_state(self, name):
            try:
                os.makedirs(name)
            except FileExistsError:
                print('Warning: A pruned_state with name='+name+' already exists. Overwriting...')

            file = open(name+'/pruned_state.txt', 'w+')
            for state in self._pruned_state:
                layer, filter = state
                file.write(str(layer)+','+str(filter)+'\n')
            file.close()
            torch.save(self.state_dict(), name+'/pruned_weights.ckpt')

        def restore_pruned_state(self, name):
            file = open(name+'/pruned_state.txt', 'r').read().strip().split('\n')
            self._pruned_state = []
            for data in file:
                layer, filter = data.strip().split(',')
                layer, filter = int(layer), int(filter)
                self.prune(layer, filter, verbose=False)
            self.load_state_dict(torch.load(name+'/pruned_weights.ckpt'))



class Datasets:

    def __init__(self, dataset, batch_size):
        self._dataset_list = ['MNIST', 'CIFAR10', 'CIFAR100', 'ImageNet', 'FakeData', 'FakeDataHighRes']
        assert dataset in self._dataset_list, 'Dataset must be in ' + ' or '.join(self._dataset_list)

        self._root = '/home/milton/Datasets/CIFAR10'

        if dataset == 'MNIST':
            self._train_dataset = torchvision.datasets.MNIST(self._root, train=True, download=False,
                                                            transform=torchvision.transforms.Compose([
                                                                torchvision.transforms.ToTensor(),
                                                                torchvision.transforms.Normalize(
                                                                    (0.1306604762738429,),
                                                                    (0.30810780717887876,)),
                                                            ]))
            self._eval_dataset = torchvision.datasets.MNIST(self._root, train=False, download=False,
                                                           transform=torchvision.transforms.Compose([
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize(
                                                                   (0.1306604762738429,),
                                                                   (0.30810780717887876,)),
                                                           ]))
        elif dataset == 'CIFAR10':
            self._train_dataset = torchvision.datasets.CIFAR10(self._root, train=True, download=False,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.RandomHorizontalFlip(),
                                                                  torchvision.transforms.RandomCrop(32, padding=4),
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(
                                                                      (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                              ]))
            self._eval_dataset = torchvision.datasets.CIFAR10(self._root, train=False, download=False,
                                                             transform=torchvision.transforms.Compose([
                                                                 torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize(
                                                                     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                             ]))
        elif dataset == 'CIFAR100':
            self._train_dataset = torchvision.datasets.CIFAR100(self._root, train=True, download=False,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.RandomHorizontalFlip(),
                                                                   torchvision.transforms.RandomCrop(32, padding=4),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize(
                                                                       (0.50707516, 0.48654887, 0.44091784),
                                                                       (0.26733429, 0.25643846, 0.27615047)),
                                                               ]))
            self._eval_dataset = torchvision.datasets.CIFAR100(self._root, train=False, download=False,
                                                              transform=torchvision.transforms.Compose([
                                                                  torchvision.transforms.ToTensor(),
                                                                  torchvision.transforms.Normalize(
                                                                      (0.50707516, 0.48654887, 0.44091784),
                                                                      (0.26733429, 0.25643846, 0.27615047)),
                                                              ]))
        elif dataset == 'ImageNet':
            self._train_dataset = torchvision.datasets.ImageNet(self._root, split='train', download=False,
                                                                transform=torchvision.transforms.Compose([
                                                                    torchvision.transforms.RandomResizedCrop(224),
                                                                    torchvision.transforms.RandomHorizontalFlip(),
                                                                    torchvision.transforms.ToTensor(),
                                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                                     (0.229, 0.224, 0.225)),
                                                               ]))
            self._eval_dataset = torchvision.datasets.ImageNet(self._root, split='val', download=False,
                                                               transform=torchvision.transforms.Compose([
                                                                   torchvision.transforms.Resize(256),
                                                                   torchvision.transforms.CenterCrop(224),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                                                                    (0.229, 0.224, 0.225)),
                                                               ]))
        elif dataset == 'FakeData':
            self._train_dataset = torchvision.datasets.FakeData(image_size=(3, 32, 32),
                                                                transform=torchvision.transforms.ToTensor())
            self._eval_dataset = torchvision.datasets.FakeData(image_size=(3, 32, 32),
                                                               transform=torchvision.transforms.ToTensor())
        elif dataset == 'FakeDataHighRes':
            self._train_dataset = torchvision.datasets.FakeData(image_size=(3, 224, 224),
                                                                transform=torchvision.transforms.ToTensor())
            self._eval_dataset = torchvision.datasets.FakeData(image_size=(3, 224, 224),
                                                               transform=torchvision.transforms.ToTensor())

        self.train_images = torch.utils.data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.eval_images = torch.utils.data.DataLoader(self._eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        self.num_train_images = self._train_dataset.__len__()
        self.num_eval_images = self._eval_dataset.__len__()

def one_hot(labels, num_class):
    return torch.cuda.FloatTensor(labels.size(0), num_class).zero_().scatter_(1, labels.unsqueeze(1), 1)








