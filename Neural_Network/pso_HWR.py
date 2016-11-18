import numpy as np
import random
from text_recog import *
from pso import *


class NN_pso(object, NN_hwr):
    """A template class for a neural network to recognise handwritten text
    Initialise with a list with each element being the number of neurons in
    that layer
    For example: neural_net = NN_hwr([400, 30, 10]) creates a neural network
    with 3 layers and 400, 30 and 10 abs their sizes"""

    def __init__(self, num_neurons_list, num_particles, exit_error=0):
        """Input must be a list of numbers."""

        for i in num_neurons_list:
            if type(i) not in [type(2)]:
                raise TypeError("Expected integer type")
        super().__init__(num_neurons_list)

        def error_function(position):
            return self.swarm_cost_function(position, X_train, y_train)

        self.training_swarm = Swarm(num_particles, error_function,
                                    exit_error=exit_error,
                                    dimension=[self.num_neurons_list[0] * self.num_neurons_list[1],
                                              self.num_neurons_list[1] * self.num_neurons_list[2],
                                              self.num_neurons_list[1], self.num_neurons_list[2]], pdeath=0.01)

    def train_nn_pso(self, X_train, y_train, n_epochs,
                     batch_size, n_iterations=100):
        """Trains the neural network with the test data. n_epochs is the number
        sweeps over whole data. batch_size is the number of training example
        per batch in the stochastic gradient descent. X_train and y_train are
        the images and labels in the training data. X_train must be a 2-D array
        with only one row and y_train is an array of length 10 of zeros
        everywhere except at the image label (where there is a 1)"""

        m = len(y_train)
        train_data = list(zip(X_train, y_train))

        for i in range(n_epochs):
            random.shuffle(train_data)
            batches = [train_data[j:j + batch_size]
                       for j in range(0, m, batch_size)]
            for batch in batches[:-1]:
                x, y = zip(*batch)
                self.train_batch_pso(x, y, n_iterations)
            print("epoch no: %d" % i, self.cost_function(X_train, y_train))

    def train_batch_pso(self, X_train, y_train, n_iterations):
        costs = self.training_swarm.optimise(n_iterations)
        position = self.training_swarm.gbest
        self.weights[0] = (np.reshape(position[0], (self.num_neurons_list[1], self.num_neurons_list[0])))
        self.weights[1] = np.reshape(position[1], (self.num_neurons_list[2], self.num_neurons_list[1]))
        self.biases[0] = np.reshape(position[2], (self.num_neurons_list[1], 1))
        self.biases[1] = np.reshape(position[3], (self.num_neurons_list[2], 1))
        return costs

    def swarm_cost_function(self, position, x, y):
        # import pdb;pdb.set_trace()
        self.weights[0] = (np.reshape(position[0], (self.num_neurons_list[1], self.num_neurons_list[0])))
        self.weights[1] = np.reshape(position[1], (self.num_neurons_list[2], self.num_neurons_list[1]))
        self.biases[0] = np.reshape(position[2], (self.num_neurons_list[1], 1))
        self.biases[1] = np.reshape(position[3], (self.num_neurons_list[2], 1))
        return self.cost_function(x, y)


if __name__ == '__main__':
    X_train, y_train = load_data("../data/traindata.mat.tar.gz")
    nn = NN_hwr([784, 15, 10], 100)
    costs = nn.train_nn_pso(X_train[:500], y_train[:500],
                            1, 500, n_iterations=10)

    accuracy = 0
    for i in range(100):
        out = nn.forward_prop(X_train[i])[0][-1]
        if np.argmax(out) == np.where(y_train[i])[0][0]:
            accuracy += 1
            print(True, np.argmax(out))
        else:
            print(False, np.argmax(out))
    print("accuracy: ", accuracy)
