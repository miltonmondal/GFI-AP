import API as api

class V:
    #number of classes
    n_c = 10
    # #pruning percentage
    # #### p = 0.60 & g_p = 0.9 provides 81% pruning
    #batch_size
    b_size = 100
    #dataset
    dataset_string = 'CIFAR10'
    #image_dim
    ##CIFAR10, CIFAR100
    image_dim = (3, 32, 32)
    # image_dim = (3, 224, 224) ##for ImageNet
    #model
    model_str = 'VGG'
    #number of layers
    n_l = 16
    ##ignore_last_few_linear layers (for resnet ig_l = 0, for vgg16 ig_l =3)
    ig_l = 3
    #restore checkpoint path for pretrained weights
    restore_checkpoint_path = '/home/milton/Pretrained_Weights/weights/vgg16-10.ckpt'
    # restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/CIFAR10/ResNet32/original_Scratch_resnet32/Training_Results_original_last_epoch/last_epoch.ckpt'

    #base path for storing results
    base_path_results = '/home/milton/DATA1/project_results/GFI_AP_results/CIFAR10_VGG16'

    #Uniform Pruning layerwise or Global Pruning (upl= 0  means global pruning)
    upl = 0
    #Pruned Normal Order (pno = 1 means first layer is pruned first, completed by last)
    #### pno = 0 means last layer is pruned first & layerwise pruning order is reversed
    pno = 1

    dataset = api.Datasets(dataset_string, batch_size=b_size)
    # def __init__(self, net, dataset):
    #     self.net = net
    #     self.dataset = dataset
