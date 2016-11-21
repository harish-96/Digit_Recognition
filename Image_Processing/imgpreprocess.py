"""
imgpreprocess.py
~~~~~~~~~~~~~~~~
A module to preprocess and segement an image containing digits in several lines
into lines and further into separate digits. These segmented images of digits 
are then fed to neural networks for recognition.
"""

import numpy as np
import os
import cv2


class Preprocess(object):

    """Preprocess class is initialized by passing an image as argument."""

    def __init__(self, imagepath):
        """:param str imagepath :  the path of the image to be segmented"""
        if type(imagepath) == str:
            if os.path.isfile(imagepath):
                try:
                    self.image = cv2.imread(imagepath, 0)
                except IOError:
                    print(imagepath, "not an image file")
            else:
                raise IOError("File doesnot exist")
        else:
            raise TypeError("Expected imagepath as a string")

    def segment_lines(self):
        """The image containing text is segmented into lines and returns
        a list of the lines
        
        :return: List of arrays, each array is a line from image

        """
        cropped_image = cropimg(binaryimg(self.image))
        lines = []
        limg = cropped_image.copy()
        last_lin = last_line(limg)
        while not np.sum(limg) == np.sum(last_lin):
            m, n = limg.shape
            p = 0
            for i in range(1, m):
                if np.sum(limg[i, :]) == 0 and np.sum(limg[i - 1, :]) != 0:
                    lines.append(limg[:i, :])
                    p = i
                    break
            limg = limg[p:, :]
            limg = cropimg(limg)
        lines.append(last_lin)
        line_len_avg = 0
        for line in lines:
            line_len_avg = line_len_avg + float(len(line)) / (2 * len(lines))
        while True:
            K = len(lines)
            for line in lines:
                if len(line) < line_len_avg:
                    lines.remove(line)
            if K == len(lines):
                break
        return lines


def binaryimg(image):
    """Converts grayscale image to binary image , it gives 1 for black and zero
    for white.

    :param array image: represents the image to be converted to binary

    """
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = np.mean(image)
    retval, binary_image = cv2.threshold(blur_image, thresh, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m, n = binary_image.shape
    binary_image = np.asmatrix(binary_image)
    for i in range(n):
        for j in range(m):
            if binary_image[j, i] == 0:
                binary_image[j, i] = 1
    for i in range(n):
        for j in range(m):
            if binary_image[j, i] == 255:
                binary_image[j, i] = 0
    return binary_image


def cropimg(image):
    """Crops a binary image tightly

    :param array image:

    """
    m, n = image.shape
    c1 = n
    c2 = 0
    r1 = m
    r2 = 0
    for i in range(n):
        for j in range(m):
            if image[j, i] == 1:
                if i < c1:
                    c1 = i
                if i > c2:
                    c2 = i
                if j < r1:
                    r1 = j
                if j > r2:
                    r2 = j
    return image[r1:r2 + 1, c1:c2 + 1]


def last_line(img):
    """Takes cropped binary image as input and gives the segment
    of image containing last line of text
    
    :param array img:

    """
    m, n = img.shape
    for i in list(reversed(range(m - 1))):
        if np.sum(img[i, :]) == 0 and np.sum(img[i + 1, :]) != 0:
            lline1 = img[i + 1:, :]
            break
    return lline1


def segment_characters(line):
    """Takes a line from a segemented image and returns a list of 
    characters in the line.
    
    :param array line:

    """
    line = cropimg(line)
    chars = []
    cimg = line.copy()
    last_character = last_char(cimg)
    while not np.sum(cimg) == np.sum(last_character):
        m, n = cimg.shape
        p = 0
        for i in range(1, n):
            if np.sum(cimg[:, i]) == 0 and np.sum(cimg[:, i - 1]) != 0:
                chars.append(cimg[:, :i])
                p = i
                break
        cimg = cimg[:, p:]
        cimg = cropimg(cimg)
    chars.append(last_character)
    return chars


def last_char(img):
    """Takes a line from a segemented image and returns last
    character in the line
    
    :param array img:

    """
    m, n = img.shape
    for i in list(reversed(range(n - 1))):
        if np.sum(img[:, i]) == 0 and np.sum(img[:, i + 1]) != 0:
            last_char = img[:, i + 1:]
            break
    return last_char
