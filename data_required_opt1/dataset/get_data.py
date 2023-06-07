import os
import numpy as np

def get_data(
data_root: str,                             # file path of data to read
    )-> tuple:

    X_train = np.expand_dims(np.load(os.path.join(data_root,'X_train.npy')), axis=1)  # load the training images (N,1,H,W)
    X_test = np.expand_dims(np.load(os.path.join(data_root,'X_test.npy')), axis=1)    # load the testing images (N,1,H,W)
    Y_train = np.load(os.path.join(data_root,'Y_train.npy'))    # load the training labels (N,)
    Y_test = np.load(os.path.join(data_root,'Y_test.npy'))      # load the testing labels (N,)

    X_train, X_test = normalize(X_train), normalize(X_test)     # normalize the pixel values to [0,1]

    return X_train, X_test, Y_train, Y_test    # return the training and testing images and labels as a tuple

def normalize(
    X: np.ndarray,                          # images to normalize (N,H,W)
    )-> np.ndarray: 

    x_max = X.max()                         # get max value of the images
    x_min = X.min()                         # get min value of the images

    return ((X-x_min)/(x_max-x_min)).astype(np.float32) # return the normalized images