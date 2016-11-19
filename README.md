## Synopsis

Given an image containing handwritten numbers, this package segments the image and identifies each digit present in the image.
Shortly, a handwritten digit recognizing software(a very crude one though). The underlying neural network can be retrained to meet specific needs

## Code Example

To train a Neural Network, use the following code snippet
~~~~~~~~~~
	from neuralnet import NN_hwr
	nn = NN_hwr([list containing number of neurons per layer])
	nn.train_nn(Input training data, label_data, Number of iterations,
	            Number of examples per batch, learning_rate)
~~~~~~~~~~
Once trained, it can be used to classify data from the command line as::
~~~~~~~~~~
    python Recognize_Digit.py path/to/image path/to/output/file (optional)
~~~~~~~~~~

## Motivation

This project has started as part of course on Python. We(Harish and Surya) formed a team and started thinking of ideas then we thought of handwritten digit recognition software using neural networks.

Recognising text has multiple applications including but not limited to Number Plate Recognition, Medical applications - scanning and documenting prescriptions, evaluating examinations etc.

## Installation

Install all the pre-requisites and run::
    python setup.py install

## API Reference

Depending on the size of the project, if it is small and simple enough the reference docs can be added to the README. For medium size to larger projects it is important to at least provide a link to where the API reference docs live.

## Tests

Describe and show how to run the tests with code examples.

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.
Harish Murali,
emailid: harish2111996@gmail.com

Surya Mohan
emailid: suryamohan1919@gmail.com, suryamohan@iitb.ac.in

## License

A short snippet describing the license (MIT, Apache, etc.)
