# Adaptive CNN filter pruning using global importance metric

## Description of functions (for GFI_AP_CIFAR & GFI_AP_IMAGENET):
1)	`variable_list`: assigns variables like dataset and model details and path for restoring pretrained models and storing results 
2)	`API`: contains all functions to modify the structure of the VGG models (backbone)
3)	`API_multi`: same as API, only difference is that it is ‘multi-gpu’ version of API, contains all functions to modify the structure of the ResNet models (backbone)
4)	`pruning_API`: contains functions related to pruning like finding normalized mean & importance score, pruning a model, retraining & finetuning
5)	`prune_main`: main script to perform pruning, retraining and finetuning
6)	`flops_API`: contains functions to compute FLOPs for a CNN
7)	`flops_main`: main script to compute FLOPs for base model and pruned model
8)	`pruning_plots_main`: main script to generate the plots to observe the effects of pruning

---

## Models support:

Given code supports pruning of filters from these models which are listed below-
```
VGG (11, 13, 16, 19)
ResNet (18, 34, 50, 101, 152, 20, 32, 44, 56, 110)
```
---

## Results: 
Some results from our paper :--

__IMAGENET_RESNET50__ results table
| Methods           | Baseline  Acc. (%) | Baseline  FLOPs (B) | Pruned  Acc. (%) | Pruned  FLOPs (B) | Acc.  drop (%) | FLOPs  reduction (%) |
|-------------------|--------------------|---------------------|------------------|-------------------|----------------|----------------------|
| Taylor-FO-BN-72%  | 76.18              | 4.09                | 74.50            | 2.25              | 1.68           | 44.99                |
| SFP               | 76.15              | 4.09                | 74.61            | 2.38              | 1.54           | 41.8                 |
| GFI-AP (p=0.3)    | 75.95              | 3.89                | 74.95            | 2.23              | 1.00           | 42.67                |

<br/>


__CIFAR10_RESNET32__ results table

| Pruning  Method | Baseline  Acc. (%) | Pruned  Acc. (%) | FLOPs  (M) | Acc.  drop (%) | FLOPs reduction (%) |
| --------------- | ------------------ | ---------------- | ---------- | -------------- | ------------------- |
| SFP             | 92.63 (± 0.7)      | 92.08 (± 0.08)   | 40.3       | 0.55           | 41.5                |
| LFPC            | 92.63 (± 0.7)      | 92.12 (± 0.3)    | 32.7       | 0.51           | 52.6                |
| GFI-AP (p=0.41) | 92.54              | 92.09 (± 0.15)   | 40.2       | 0.45           | 42.5                |

<br/>



## Sample checkpoints:
Checkpoints can be found in this google drive folder:
https://drive.google.com/drive/folders/1g45M7zaaeXvCQWscoQex33U5flk9Zg4F?usp=sharing

1)	pretrained_checkpoints: contains baseline checkpoints

2)	checkpoints_after_pruning: contains checkpoints after pruning and fine-tuning

---

### Citation:

---

### Acknowledgement:


__FLOPs counter code__: https://github.com/sovrasov/flops-counter.pytorch
