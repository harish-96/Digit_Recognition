import numpy as np
import struct
import random


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
            delta_b , delta_w = self.backprop(x,y)
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
        return zip(image_input, labels)
