import unittest
import math
import numpy as np
from NeuralNet import *


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        X_train, y_train = load_data("./traindata.mat")
        self.nn = NN_hwr([len(X_train[0]), 30, 10])

    def test_NN_hwr_raises_exception_for_non_numeric_values(self):
        self.assertRaises(TypeError, NN_hwr, ["sb", "uir", 5])

    def test_sigmoid_function_returns_correct_value(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 1 / (1 + e**-1))
        self.assertListEqual(sigmoid([0, 0, 0], np.array([0.5, 0.5, 0.5])))

    def test_sigmoid_derivative_returns_correct_value(self):
        self.assertAlmostEqual(sigmoid_derivative(0), 0.25)
        self.assertAlmostEqual(sigmoid(1), 0.19661193324)
        self.assertAlmostEqual(sigmoid(36), 0.0)
        self.assertAlmostEqual(sigmoid(-36), 0.0)

    def test_back_prop_rejects_inputs_of_incorrect_sizes(self):
        self.assertRaises(TypeError, nn.back_prop, [np.zeros((782, 1)), np.zeros((1, 10))])
        self.assertRaises(TypeError, nn.back_prop, ["abc", "def"])


if __name__ == '__main__':
    unittest.main()
