import numpy as np
import struct
import random
import matplotlib.pyplot as plt


class HWR(object):

    def __init__(self, num_neurons_list):
   	"""  -------------  """
        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.biases = [np.random.randn(y, 1) for y in num_neurons_list[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(num_neurons_list[:-1], num_neurons_list[1:])]
    def train_using_SGD(self,train_data , n_sweeps , batch_size , learning_rate ):
        """------------------"""
   	n = len(train_data)
	for j in xrange(n_sweeps):
		random.shuffle(train_data)
		batches = [train_data[j:j+batch_size] for j in xrange(0,n,batch_size)]
		for batch in batches :
			self.update_batch(batch , learning_rate)
    def update_batch(self,batch , learning_rate):
        """--------------------"""
        total_delta_b = [np.zeros(b.shape) for b in self.biases]
        total_delta_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            delta_b , delta_w = self.back_propagation(x,y)
            total_delta_b = total_delta_b + delta_b
            total_delta_w = total_delta_w + delta_w
        self.biases = self.biases-((learning_rate/len(batch))*total_delta_b)
        self.weights = self.weights-((learning_rate/len(batch))*total_delta_w)
    def feedforward(self, a):
        """--------------------"""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_function(np.dot(w, a)+b)
        return a
    def back_propagation(self,x,y):
        """---------------------"""
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activated_neuron = x
        activated_neurons = [x]
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activated_neuron)+b
            zs.append(z)
            activated_neuron = sigmoid_function(z)
            activated_neurons.append(activated_neuron)
        delta = self.cost_derivative(activated_neurons[-1], y)*sigmoid_derivative(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activated_neurons[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activated_neurons[-l-1].transpose())
        return (delta_b, delta_w)
    def cost_derivative(self, output_activations, y):
        """-------------------------"""
        return (output_activations-y)

def sigmoid_function(t):
    """-------------------"""
    return 1.0/(1.0+np.exp(-t))
def sigmoid_derivative(t):
    """"----------------"""
    return sigmoid_function(t)*(1-sigmoid_function(t))
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
        image_input = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)
        return image_input, labels


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


X_train, y_train = unpack_dat("./train-images.idx3-ubyte", "./train-labels.idx1-ubyte")
display_data(X_train[:10], 2, 5)

nn = HWR([len(X_train[0]), 15, 10])
nn.train_using_SGD(zip(X_train[1:10], y_train[1:10]), 100, 1, 0.01)
# print(nn.feedforward(X_train[5])[0][-1])
# print("expected output is %d" % (y_train[5]))