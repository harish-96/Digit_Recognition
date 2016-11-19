from Neural_Network import neuralnet as nln
import numpy as np
import scipy.io as sio
def unpack_dat(imgpath, labpath):
    """ Unpack images and labels obtained online from
    `MNIST Database <http://yann.lecun.com/exdb/mnist/>`_
    """
    with open(labpath, 'rb') as f:
        magic_no, n_dim = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(imgpath, 'rb') as f:
        magic_num, n_dim, n_rows, n_cols = struct.unpack(">iiii",
                                                         f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)

    images = [np.reshape(x, (784, 1)) for x in images]
    labels = [np.array([y == i for i in range(10)])[np.newaxis].T
              for y in labels]

    return images, labels

def load_data(path):
    """Loads the image data from the path provided and returns the images and
    labels

    :param string path: The path to the file where the training/test data is
    present

    :return: A dictionary object containing the input data and labels
    Keys of the dict object -- 'X_train' and 'y_train' or 'X_test' and 'y_test'

    """
    if os.path.splitext(path)[1] == '.gz':
        tfile = tarfile.open(path)
        tfile.extractall("data/")
        tfile.close()
        path = os.path.splitext(os.path.splitext(path)[0])[0]
    data_dict = sio.loadmat(path)
    if 'train' in path:
        return data_dict['X_train'], data_dict['y_train']
    else:
        return data_dict['X_test'], data_dict['y_test']
def display_data(imgs, nrows=1, ncols=1, nx_pixels=28, ny_pixels=28):
    """Dispay the images given in X.

    :param int nrows: The number of plots per column. The default value is 1
    :param int ncols: The number of plots per row. The default value is 1
    :param int nx_pixels: The number of pixels along axis 1. The default value is 28
    :param int ny_pixels: The number of pixels along axis 2. The default value is 28

    :return: None.\ Displays the Image Data

    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)

    if (nrows + ncols) == 2:
        ax.imshow(imgs[0].reshape(nx_pixels, ny_pixels),
                  cmap='Greys', interpolation="bicubic")
    else:
        ax = ax.flatten()
        for i in range(nrows * ncols):
            ax[i].imshow(imgs[i].reshape(nx_pixels, ny_pixels),
                         cmap='Greys', interpolation="bicubic")
    plt.tight_layout()
    plt.show()

X_train, y_train = load_data("data/traindata.mat.tar.gz")
X_test, y_test = load_data("data/testdata.mat.tar.gz")


nn = nln.NN_hwr([len(X_train[0]), 20, 40, 20, 10])
nn.train_nn(X_train, y_train, 10, 20, 0.06)

accuracy = 0
for i in range(len(X_test[:100])):
    out = nn.forward_prop(X_test[i])[0][-1]
    if np.argmax(out) == np.where(y_test[i])[0][0]:
        accuracy += 1
        print(True, np.argmax(out))
    else:
        print(False, np.argmax(out))
sio.savemat("../data/weights_biases",{'w':nn.weights, 'b':nn.biases})
print("accuracy: ", accuracy)
