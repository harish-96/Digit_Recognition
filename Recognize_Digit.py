from Neural_Network.neuralnet import NN_hwr
from Neural_Network.train_network import load_data
import Image_Processing.imgpreprocess as igp
from Image_Processing.center_image import *
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys


dat = sio.loadmat("data/weights_biases.mat")
X_train, y_train = load_data("data/traindata.mat.tar.gz")
X_test, y_test = load_data("data/testdata.mat.tar.gz")

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
n = 0

for line in lines:
    n += 1
    numbers = "Line no: " + str(n) + " : "
    for char in igp.segment_characters(line):
        chars.append(char)
        char028 = np.zeros((28, 28))
        image = Image.fromarray(char)
        char0 = np.array(image.resize((20, 20), Image.ANTIALIAS))
        for i in range(20):
            for j in range(20):
                char028[4 + i][4 + j] = char0[i][j]
        char028 = center_image(char028)
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
