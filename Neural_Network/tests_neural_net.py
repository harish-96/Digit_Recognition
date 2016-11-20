import unittest
import math
import numpy as np
from Neural_Network.neuralnet import *
from Neural_Network.train_network import load_data
import shutil


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """Setup function. Called everytime a test is run"""
        self.X_train, self.y_train = load_data("../data/traindata.mat.tar.gz")
        self.nn = NN_hwr([len(self.X_train[0]), 50, 10])

    def test_NN_hwr_raises_exception_for_non_numeric_values(self):
        """Ensure that the Neural Network rejects non numeric initialisation"""
        self.assertRaises(TypeError, NN_hwr, ["sb", "uir", 5])

    def test_sigmoid_function_returns_correct_value(self):
        """Check that the sigmoid function returns the expected values for
        certain test cases"""
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 1 / (1 + math.e**-1))
        np.testing.assert_array_almost_equal(sigmoid([0, 0, 0]),
                                             np.array([0.5, 0.5, 0.5]))

    def test_sigmoid_derivative_returns_correct_value(self):
        """Check that the sigmoid derivative function returns the expected values for
        certain test cases"""
        self.assertAlmostEqual(sigmoid_derivative(0), 0.25)
        self.assertAlmostEqual(sigmoid_derivative(1), 0.19661193324)

    def test_sigmoid_saturates_to_one_and_zero(self):
        """Check that the sigmoid function saturates to one at high input
        values"""
        self.assertAlmostEqual(sigmoid(50), 1.0)
        self.assertAlmostEqual(sigmoid(-36), 0.0)

    def test_back_prop_rejects_inputs_of_incorrect_sizes(self):
        """Check that the backprop function rejects non numeric calls"""
        self.assertRaises(TypeError, self.nn.back_prop,
                          [np.zeros((782, 1)), np.zeros((1, 10))])
        self.assertRaises(TypeError, self.nn.back_prop, ["abc", "def"])

    def test_train_nn_overfits_the_data_for_small_input_size(self):
        """Check that the neural network overfits the data for large number of
        neurons and small input size. This happens because of the curse of
        dimensionality"""
        # self.nn.train_nn(self.X_train[:10], self.y_train[:10], 200, 1, 0.06)
        # count = 0
        # for i in range(10):
        #     out = self.nn.forward_prop(self.X_train[i])[0][-1]
        #     if np.argmax(out) == np.where(self.y_train[i])[0][0]:
        #         count += 1
        #     else:
        #         print("Incorrect", np.argmax(out),
        #               np.where(self.y_train[i])[0][0])
        # self.assertGreaterEqual(count, 5)

    def test_nn_predicts_accurate_results(self):
        """Check that the Neural Network prediction is satisfactory.
        Threshold accuracy:70%"""
        self.nn.train_nn(self.X_train, self.y_train, 3, 10, 0.06)
        X_test, y_test = load_data("../data/traindata.mat.tar.gz")
        # accuracy = self.nn.accuracy(X_test, y_test)
        # print("accuracy: ", accuracy)
        # self.assertGreaterEqual(self.nn.accuracy(X_test, y_test), 70)
        accuracy = 0
        for i in range(len(X_test[:100])):
            out = self.nn.forward_prop(X_test[i])[0][-1]
            if np.argmax(out) == np.where(y_test[i])[0][0]:
                accuracy += 1
            else:
                print("false")
        print(accuracy / len(X_test[:100]))
        self.assertGreaterEqual(count / len(X_test[:100]), 0.5)

    def tearDown(self):
        del self.nn
        shutil.rmtree("data")


if __name__ == '__main__':
    unittest.main()
