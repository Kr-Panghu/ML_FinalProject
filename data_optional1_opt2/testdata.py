import os
import numpy as np
from matplotlib import pyplot as plt

from dataset import get_data,normalize
if __name__ == '__main__':
    ######################## Get train dataset ########################
    X_train = get_data('dataset')
    ########################################################################
    ######################## Implement you code here #######################
    ########################################################################
    
print(X_train)
print("shape:", X_train.shape)