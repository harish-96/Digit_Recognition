import scipy.io as sio
from Neural_Network.text_recog import *
import Image_Processing.imgpreprocess as igp
import numpy as np
from Image_Processing.center_image import *
from PIL import Image


dat = sio.loadmat("../data/weights_biases.mat")
X_train, y_train = load_data("../data/traindata.mat.tar.gz")
X_test, y_test = load_data("../data/testdata.mat.tar.gz")
# display_data(X_train[:10], 2, 5)

nn = NN_hwr([len(X_train[0]), 15, 10])
nn.weights = dat['w'][0]
nn.biases = dat['b'][0]

k = igp.Preprocess("im.jpg")
lines = k.segment_lines()
chars = []
for line in lines:
    for char in igp.segment_characters(line):
        chars.append(char)

for i in chars:
    char028 = np.zeros((28, 28))
    image = Image.fromarray(i)
    # char0 = cv2.resize(i, (20, 20))
    char0 = np.array(image.resize((20, 20), Image.ANTIALIAS))
    for i in range(20):
        for j in range(20):
            char028[4 + i][4 + j] = char0[i][j]
    char028 = center_image(char028)
    plt.imshow(char028)
    plt.show()
    char028re = np.reshape(char028, (784, 1))
    ex = nn.forward_prop(char028re)[0][-1]
    print(np.argmax(ex))
