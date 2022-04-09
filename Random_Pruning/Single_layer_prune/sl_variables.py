import API as api

class V:
    #number of classes
    n_c = 100
    #batch_size
    b_size = 100
    #dataset
    dataset_string = 'CIFAR100'
    #image_dim
    ##CIFAR10, CIFAR100
    image_dim = (3, 32, 32)
    #model
    model_str = 'VGG'
    #number of layers
    n_l = 16
    ##ignore_last_few_linear layers (for resnet ig_l = 0, for vgg16 ig_l =3)
    ig_l = 3
    #restore checkpoint path for pretrained weights
    restore_checkpoint_path = '/home/milton/Pretrained_Weights/weights/vgg16-100.ckpt'

    #base path for storing results
    base_path_results = '/home/milton/DATA1/project_results/random_prune_results/single_layer_pruning/CIFAR100_VGG16'

    dataset = api.Datasets(dataset_string, batch_size=b_size)
    # def __init__(self, net, dataset):
    #     self.net = net
    #     self.dataset = dataset
