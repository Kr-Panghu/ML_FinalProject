import os
import numpy as np
from dataset import get_data
from matplotlib import pyplot as plt
if __name__ == '__main__':
######################## Get train/test dataset ########################
    X_train,X_test,Y_train,Y_test = get_data('dataset')

# Convert 0 to -1 in Y_train
Y_train = 2*(Y_train.astype(int)-1) + 1

# Convert 0 to -1 in Y_test
Y_test = 2*(Y_test.astype(int)-1) + 1


(60000, 1, 32, 32)
print("X_train:", X_train)
print("X_train_shape:", X_train.shape)

(60000, )
print("Y_train:", Y_train)
print("Y_train_shape:", Y_train.shape)

(10000, 1, 32, 32)
print("X_test:", X_test)
print("X_test_shape:", X_test.shape)

(10000, )
print("Y_test:", Y_test)
print("Y_test_shape:", Y_test.shape)