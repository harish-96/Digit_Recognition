import Image_Processing.imgpreprocess as igp
import cv2
import numpy as np
from Image_Processing.center_image import *
from Neural_Network.text_recog import *


def train_net():
    X_train, y_train = load_data("../data/traindata.mat.tar.gz")
    X_test, y_test = load_data("../data/testdata.mat.tar.gz")
    display_data(X_train[:10], 2, 5)

    nn = NN_hwr([len(X_train[0]), 15, 10])
    nn.train_nn(X_train, y_train, 10, 20, 0.06)

    accuracy = 0
    for i in range(len(X_test[:100])):
        out = nn.forward_prop(X_test[i])[0][-1]
        if np.argmax(out) == np.where(y_test[i])[0][0]:
            accuracy += 1
            print(True, np.argmax(out))
        else:
            print(False, np.argmax(out))
    print("accuracy: ", accuracy)
    return nn


nn = train_net()
k = igp.Preprocess("testimage.jpg")
lines = k.segment_lines()
chars = []
for line in lines:
    for char in igp.segment_characters(line):
        chars.append(char)

for i in chars:
    char028 = np.zeros((28, 28))
    char0 = cv2.resize(i, (20, 20))
    for i in range(20):
        for j in range(20):
            char028[4 + i][4 + j] = char0[i][j]
    char028 = center_image(char028)
    plt.imshow(char028)
    plt.show()
    char028re = np.reshape(char028, (784, 1))
    ex = nn.forward_prop(char028re)[0][-1]
    print(np.argmax(ex))
