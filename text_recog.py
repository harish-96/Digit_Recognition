import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import tarfile


def unpack_dat(imgpath, labpath):
    """ Unpack images and labels obtained online from
    http://yann.lecun.com/exdb/mnist/
    """
    with open(labpath, 'rb') as f:
        magic_no, n_dim = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(imgpath, 'rb') as f:
        magic_num, n_dim, n_rows, n_cols = struct.unpack(">iiii",
                                                         f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)
    images = [np.reshape(x, (784, 1)) for x in images]
    labels = [np.array([y == i for i in range(10)])[np.newaxis].T for y in labels]
    return images, labels


def display_data(imgs, nrows=1, ncols=1, nx_pixels=28, ny_pixels=28):
    """Dispay the images given in X. 'nrows' and 'ncols' are
    the number of rows and columns in the displayed data"""
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


def sigmoid(z):
    """-------------------"""
    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NN_hwr(object):
    """A template class for a neural network to recognise handwritten text"""

    def __init__(self, num_neurons_list):
        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.biases = [np.random.randn(y, 1) for y in num_neurons_list[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(num_neurons_list[:-1], num_neurons_list[1:])]

    def forward_prop(self, x_train):
        activations = []
        z = []
        activations.append(x_train)
        for i in range(self.num_layers - 1):
            z.append(np.dot(self.weights[i], activations[-1]) + self.biases[i])
            activations.append(sigmoid(z[-1]))
        return activations[1:], z

    # def back_prop(self, training_example):
    #     """training_example is a tuple with element one an np array and element 2 a scalar"""
    #     x_train, y_train = training_example
    #     activations, z = self.forward_prop(x_train)
    #     delta_b = np.array(self.biases[:])
    #     delta_w = np.array(self.weights[:])
    #     delta_L = self.cost_derivative(activations[-1], y_train) * sigmoid_derivative(z[-1])

    #     delta_b[-1] = delta_L
    #     delta_w[-1] = np.outer(delta_L, activations[-2])
    #     delta = delta_L
    #     for i in range(2, self.num_layers - 1):
    #         sd = sigmoid_derivative(z[-i])
    #         delta = np.outer(self.weights[-i + 1].transpose(), delta) * sd
    #         delta_b[-i] = delta
    #         if i == 1:
    #             ac = x_train
    #         else:
    #             ac = activations[-i - 1]
    #         delta_w[-i] = np.outer(delta, ac)
    #     return (delta_b, delta_w)

    def back_prop(self, training_example):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        x, y = training_example
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)

    def train_batch(self, batch, learning_rate):
        """ batch is a list of tuples with first element being a numpy array and second a scalar"""
        delta_b_sum = [np.zeros(b.shape) for b in self.biases]
        delta_w_sum = [np.zeros(w.shape) for w in self.weights]
        for training_example in batch:
            delta_b, delta_w = self.back_prop(training_example)
            # delta_b_sum += delta_b
            # delta_w_sum += delta_w
            delta_b_sum = [db + ddb for db, ddb in zip(delta_b_sum, delta_b)]
            delta_w_sum = [nw + dnw for nw, dnw in zip(delta_w_sum, delta_w)]
        # self.biases = self.biases - learning_rate * delta_b_sum
        # self.weights = self.weights - learning_rate * delta_w_sum
        self.weights = [w - (learning_rate / len(batch)) * nw
                        for w, nw in zip(self.weights, delta_w_sum)]
        self.biases = [b - (learning_rate / len(batch)) * nb
                       for b, nb in zip(self.biases, delta_b_sum)]

    def train_nn(self, X_train, y_train, n_epochs, batch_size, learning_rate):
        m = len(y_train)
        train_data = list(zip(X_train, y_train))

        for i in range(n_epochs):
            random.shuffle(train_data)
            batches = [train_data[j:j + batch_size]
                       for j in range(0, m, batch_size)]
            for batch in batches:
                self.train_batch(batch, learning_rate)
            print("epoch no: %d" % i, self.cost_function(X_train, y_train))

    def cost_function(self, X_train, y_train):
        pass
    #     J = 0
    #     for i in range(len(y_train)):
    #         J += 0.5 * np.sum((self.forward_prop(X_train[i])[0][-1] - y_train)**2)
    #     return J

    def cost_derivative(self, activation, y):
        return np.array(activation) - y


def load_data(path):
    tfile = tarfile.open(path + ".tar.gz", 'r:gz')
    tfile.extractall(".")
    tfile.close()
    data_dict = sio.loadmat(path)
    return data_dict['X_train'], data_dict['y_train']


X_train, y_train = load_data("./traindata.mat")
display_data(X_train[:10], 2, 5)

nn = NN_hwr([len(X_train[0]), 15, 10])
nn.train_nn(X_train, y_train, 5, 20, 0.03)

accuracy = 0
for i in range(20, 40):
    out = nn.forward_prop(X_train[i])[0][-1]
    if np.argmax(out) == np.where(y_train[i])[0][0]:
        accuracy += 1
        print(True, np.argmax(out))
    else:
        print(False, np.argmax(out))
print("accuracy: ", accuracy)
display_data(X_train[20:40], 4, 5)
