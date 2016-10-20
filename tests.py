import unittest
import math
import numpy as np
from text_recog import *


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.X_train, self.y_train = load_data("./traindata.mat.tar.gz")
        self.nn = NN_hwr([len(self.X_train[0]), 50, 10])

    def test_NN_hwr_raises_exception_for_non_numeric_values(self):
        self.assertRaises(TypeError, NN_hwr, ["sb", "uir", 5])

    def test_sigmoid_function_returns_correct_value(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 1 / (1 + math.e**-1))
        np.testing.assert_array_almost_equal(sigmoid([0, 0, 0]), np.array([0.5, 0.5, 0.5]))

    def test_sigmoid_derivative_returns_correct_value(self):
        self.assertAlmostEqual(sigmoid_derivative(0), 0.25)
        self.assertAlmostEqual(sigmoid_derivative(1), 0.19661193324)

    def test_sigmoid_saturates_to_one_and_zero(self):
        self.assertAlmostEqual(sigmoid(50), 1.0)
        self.assertAlmostEqual(sigmoid(-36), 0.0)

    def test_back_prop_rejects_inputs_of_incorrect_sizes(self):
        self.assertRaises(TypeError, self.nn.back_prop, [np.zeros((782, 1)), np.zeros((1, 10))])
        self.assertRaises(TypeError, self.nn.back_prop, ["abc", "def"])

    def test_train_nn_overfits_the_data_for_small_input_size(self):
        self.nn.train_nn(self.X_train[:10], self.y_train[:10], 200, 1, 0.06)
        count = 0
        for i in range(10):
            out = self.nn.forward_prop(self.X_train[i])[0][-1]
            if np.argmax(out) == np.where(self.y_train[i])[0][0]:
                count += 1
        self.assertGreater(count, 8)

    def tearDown(self):
        del self.nn


if __name__ == '__main__':
    unittest.main()
