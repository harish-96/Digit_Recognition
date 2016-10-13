import numpy as np
import scipy
import os
import struct


def unpack_dat(imgpath, labpath):
    """ Unpack images and labels obtained online from 
    http://yann.lecun.com/exdb/mnist/
    """
    with open(labpath, 'rb') as f:
        magic_no, n_dim = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype = np.uint8)
        
    with open(imgpath, 'rb') as f:
        magic_num, n_dim, n_rows, n_cols = struct.unpack(">iiii", 
                                           f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)
        return images, labels


X_train, y_train = unpack_dat("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")

