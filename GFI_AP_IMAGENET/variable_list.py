import API_multi as api

class V:
    #number of classes
    n_c = 1000 #200
    # #pruning percentage
    # #### p = 0.60 & g_p = 0.9 provides 81% pruning
    #batch_size
    b_size = 128 #64
    #dataset
    dataset_string ='ImageNet' #'TinyImageNet'
    #image_dim
    ##(3, 32, 32) CIFAR10, CIFAR100
    image_dim = (3, 224, 224) #(3,64,64)
    # image_dim = (3, 224, 224) ##for ImageNet
    #model
    model_str = 'ResNet'
    #number of layers
    n_l = 50 #18
    ##ignore_last_few_linear layers (for resnet ig_l = 0, for vgg16 ig_l =3)
    ig_l = 0
    #restore checkpoint path for pretrained weights
    restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/ImageNet/ResNet50/original_Scratch_resnet50/Training_Results_original_best_epoch/best_epoch.ckpt'
    # restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/TinyImageNet/ResNet18/original_Scratch_resnet18/Training_Results_original_best_epoch/best_epoch.ckpt'
    # restore_checkpoint_path = '/home/milton/DATA1/project_results/GFI_AP_input_ckpt/CIFAR10/ResNet32/original_Scratch_resnet32/Training_Results_original_last_epoch/last_epoch.ckpt'

    #base path for storing results
    base_path_results = '/home/milton/DATA1/project_results/GFI_AP_results/ImageNet_ResNet50'

    #Uniform Pruning layerwise or Global Pruning (upl= 0  means global pruning)
    upl = 0
    #Pruned Normal Order (pno = 1 means first layer is pruned first, completed by last)
    #### pno = 0 means last layer is pruned first & layerwise pruning order is reversed
    pno = 1

    dataset = api.Datasets(dataset_string, b_size)
    # def __init__(self, net, dataset):
    #     self.net = net
    #     self.dataset = dataset
