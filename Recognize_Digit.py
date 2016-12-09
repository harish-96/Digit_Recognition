from Neural_Network.neuralnet import NN_hwr
from Neural_Network.train_network import load_data
import Image_Processing.imgpreprocess as igp
from Image_Processing.center_image import *
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy.io as sio


pathrecog = sys.argv[0]
pathrecog = os.path.abspath(pathrecog)
pathrecog = pathrecog[:-18]
dat = sio.loadmat(pathrecog + "data/weights_biases.mat")
X_train, y_train = load_data(pathrecog + "data/traindata.mat.tar.gz")
X_test, y_test = load_data(pathrecog + "data/testdata.mat.tar.gz")

nn = NN_hwr([len(X_train[0]), 15, 10])
nn.weights = dat['w'][0]
nn.biases = dat['b'][0]

path = sys.argv[1]
output = 0
if len(sys.argv) == 3:
    output = 1
    outfile_path = sys.argv[2]
    if os.path.exists(outfile_path):
        os.remove(outfile_path)

k = igp.Preprocess(path)
lines = k.segment_lines()
chars = []
n_line = 0

for line in lines:
    n_line += 1
    numbers = "Line no: " + str(n_line) + " : "
    for char in igp.segment_characters(line):
        chars.append(char)
        char028 = center_image(char)
        char028re = np.reshape(char028, (784, 1))
        ex = nn.forward_prop(char028re)[0][-1]
        if output:
            numbers = numbers + str(np.argmax(ex))
        else:
            plt.imshow(char028)
            plt.show()
            print(np.argmax(ex))
    if output:
        with open(outfile_path, "a") as outfile:
            outfile.write("\n" + numbers)
