import os
import numpy as np

def get_data(
    data_root: str,                             # file path of data to read
        )-> np.ndarray:

    X_train = np.load(os.path.join(data_root,'X_train.npy'))    # load the training images (N,3,H,W)

    X_train = normalize(X_train)            # normalize the pixel values to [0,1]

    return X_train                          # return the training images

def normalize(
    X: np.ndarray,                          # images to normalize (N,3,H,W)
    )-> np.ndarray: 

    x_max = X.max()                         # get max value of the images
    x_min = X.min()                         # get min value of the images

    return ((X-x_min)/(x_max-x_min)).astype(np.float32) # return the normalized images