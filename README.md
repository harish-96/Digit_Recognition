## Synopsis

Given an image containing handwritten numbers, this package segments the image and identifies each digit present in the image.
Shortly, a handwritten digit recognizing software(a very crude one though). The underlying neural network can be retrained to meet specific needs
##Documentation

To access the documentation in Github Pages, [Click here](https://harish-96.github.io/Digit_Recognition/)

## Code Example

To train the Neural Network, use 
    python path/to/train_network.py 

If you are in the project directory use
    python Neural_Network/train_network.py

To recognize digits from image use as :
    python path/to/Recognize_Digit.py path/to/image path/to/outputfile

If you are in project directory and to test on the default test image use as:
    python Recognize_Digit.py testimage.jpg output.txt
You can find the result in output.txt

## Motivation

Recognising text has multiple applications including but not limited to Number Plate Recognition, Medical applications - scanning and documenting prescriptions, evaluating examinations etc.

This project has started as part of course on Software Development.
## Installation

The pre-requisites for this software are

1. Numpy
2. Scipy
3. PIL
4. OpenCV

Install all the pre-requisites and run::
    python setup.py install

## Contributors

Let people know how they can dive into the project, include important links to things like issue trackers, irc, twitter accounts if applicable.

Harish Murali, 
email-id: harish2111996@gmail.com

Surya Mohan, 
email-id: suryamohan1919@gmail.com, suryamohan@iitb.ac.in

