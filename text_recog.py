import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import pdb
import scipy.io as sio

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
        return images, labels


def display_data(imgs, nrows, ncols, nx_pixels=28, ny_pixels=28):
    """Dispay the images given in X. 'nrows' and 'ncols' are
    the number of rows and columns in the displayed data"""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)
    ax = ax.flatten()

    for i in range(nrows * ncols):
        ax[i].imshow(imgs[i].reshape(nx_pixels, ny_pixels),
                     cmap='Greys', interpolation="bicubic")
    plt.tight_layout()
    plt.show()


# def sigmoid(z):
#     z = np.asarray(z, dtype='float')
#     shape = np.shape(z)
#     z.flatten()
#     for i in range(len(z)):
#         z[i] = 1 / (1 + math.e**-z[i])
#     return z.reshape(shape)

def sigmoid(z):
    """-------------------"""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class NN_hwr(object):
    """A template class for a neural network to recognise handwritten text"""

    def __init__(self, num_neurons_list):
        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.biases = [[random.random() for i in range(self.num_neurons_list[j])] for j in range(1, self.num_layers)]
        self.biases = np.array(self.biases)
        self.weights = [np.array([[random.random() for i in range(self.num_neurons_list[x])]
                        for j in range(self.num_neurons_list[x + 1])])
                        for x in range(self.num_layers - 1)]
    def forward_prop(self, x_train):
        """ For a single training example"""
        z = np.zeros_like(self.biases) # biases does not include layer 1 and neither does z
        activations = np.zeros_like(self.biases) # Not including the first layer.
        z[0] = np.dot(self.weights[0], x_train) + self.biases[0]
        activations[0] = sigmoid(z[0])
        
        for i in range(1, self.num_layers - 1):
            z[i] = np.dot(self.weights[i],
                            activations[i - 1]) + self.biases[i])
            activations[i] = sigmoid(z[i])
        return activations, z

    def back_prop(self, training_example):
        """training_example is a tuple with element one an np array and element 2 a scalar"""
        x_train, y_train = training_example
        activations, z = self.forward_prop(x_train)
        y_train_ar = [y_train == i for i in range(10)]
        #delta_L = np.array([self.cost_derivative(activations[-1],
        #                               y_train_ar)[i] * sigmoid_derivative(z[-1])[i] for i in range(len(z[-1]))])
        delta_L = self.cost_derivative(activations[-1], y_train_ar).transpose() * sigmoid_derivative(z[-1])
        #print(len(activations[-1]), activations[-1])
        delta_b = np.zeros_like(self.biases)
        delta_w = np.zeros_like(self.weights)

        delta_b[-1] = np.array(delta_L)
        ac = np.array([[i] for i in activations[-2]])
        delta_w[-1] = np.dot(delta_L.transpose(), ac.transpose())
        delta = delta_L
        for i in range(2, self.num_layers - 1):
            sd = sigmoid_derivative(z[-i])
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sd
            delta_b[-i] = delta
            ac = np.array([[j] for j in activations[-i]])
            delta_w[-i] = np.dot(delta, ac.transpose())
        return (delta_b, delta_w)

    def train_batch(self, batch, learning_rate):
        """ batch is a list of tuples with first element being a numpy array and second a scalar"""
        # delta_b_sum = np.array([np.zeros(len(b)) for b in self.biases])
        #delta_w_sum = np.array([np.zeros(w.shape) for w in self.weights])
        delta_b_sum = np.zeros_like(self.biases)
        delta_w_sum = np.zeros_like(self.weights)
        for training_example in batch:
            delta_b, delta_w = self.back_prop(training_example)
            delta_b_sum += delta_b
            delta_w_sum += delta_w
            del delta_b, delta_w
        self.biases = self.biases - learning_rate * delta_b_sum
        self.weights = self.weights - learning_rate * delta_w_sum

    def train_nn(self, X_train, y_train, n_epochs, batch_size, learning_rate):
        m = len(y_train)
        train_data = list(zip(X_train, y_train))

        for i in range(n_epochs):
            np.random.shuffle(train_data)
            batches = [train_data[j:j + batch_size]
                       for j in range(0, m, batch_size)]
            count = 0
            count1 = 0
            #self.train_batch(batches[0], learning_rate)
            #self.train_batch(batches[1], learning_rate)
            for batch in batches:
                count +=1;print("Batch no: %d" % count)
                print(batch)
                self.train_batch(batch, learning_rate)
            # print(self.cost_function(X_train, y_train))

    def cost_function(self, X_train, y_train):
        J = 0
        for i in range(len(y_train)):
            y_train_ar = [y_train[i] == j for j in range(10)]
            J += 0.5 * np.sum((nn.forward_prop(X_train[i])[0][-1] - y_train_ar)**2)
        return J

    def cost_derivative(self, activation, y):
        global count1
        count1 = count1 + 1
#        if len(activation) != len(y):
#            print("The count is ", count1)
#            #print(activation)
#            #pdb.set_trace()
#            return np.array([activation[0][i] - y[i] for i in range(len(y))])
        return np.array([activation[i] - y[i] for i in range(len(y))])

X_train, y_train = unpack_dat("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
display_data(X_train[:10], 2, 5)
#X_train = (X_train - 128) / 256
mat_data = sio.loadmat("/home/harish/Dropbox/machine-learning-ex4/ex4/ex4data1.mat")
X1 = mat_data['X']
y1 = mat_data['y'][:,0]
count1 = 0
nn = NN_hwr([len(X1[0]), 15, 10])
nn.train_nn(X1[500:1500], y1[500:1500], 10, len(y1[500:1500]), 0.01)
out = nn.forward_prop(X1[550])[0][-1][0]
print(np.argmax(out))
print("expected output is %d" % (y1[550]))
