import numpy as np
import math
import scipy
import os
import struct
import matplotlib.pyplot as plt


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


def display_data(imgs, nrows, ncols, nx_pixels=28, ny_pixels=28):
    """Dispay the images given in X. 'nrows' and 'ncols' are
    the number of rows and columns in the displayed data"""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)
    ax = ax.flatten()

    for i in range(nrows*ncols):
        ax[i].imshow(imgs[i].reshape(nx_pixels,ny_pixels), cmap='Greys', interpolation="bicubic")
    plt.tight_layout()
    plt.show()


def cost_function(params, num_labels, input_size, hl1_size, X, y):
    """Returns the cost function for the neural network and its gradient given the
    network parameters(params), number of labels(num_labels) in the final output,
    size of input layer(input_size), size of hidden layer(hl1_size) and the
    training data(X) with labels(y)"""

    theta1 = reshape(params[0:hl1_size * (input_size + 1)], hl1_size, (input_size + 1));
    theta2 = reshape(params[(1 + (hl1_size * (input_size + 1))):], num_labels, (hl1_size + 1)); 

    m = len(X) # Number of training examples

X_train, y_train = unpack_dat("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
display_data(X_train[:10], 2, 5)
