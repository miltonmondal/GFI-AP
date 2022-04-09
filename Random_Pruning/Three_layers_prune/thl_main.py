import torch
import os
import csv
import numpy as np
import API as api
from thl_variables import V
import thl_function as func

trials = 5 ## number of trilas
prune_factor = 0.99  ## pruning fraction of filters from each layer

### for retraining
epochs = 12
lear_change_freq = 3
lear1 = 0.001
lr_divide_factor = 3
lr_change_freq = 3

os.makedirs(V.base_path_results, exist_ok=True)

for t in range(4,trials+1):
    func.thl_prune(prune_factor, t, epochs, lear_change_freq, lear1, lr_divide_factor, lr_change_freq)
