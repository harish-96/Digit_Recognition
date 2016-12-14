"""
imgpreprocess.py
~~~~~~~~~~~~~~~~
A module to preprocess and segement an image containing digits in several lines
into lines and further into separate digits. These segmented images of digits
are then fed to neural networks for recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import Image_Processing.center_image as c_img


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

    def segment_lines(self, bin_block_size=0.5):
        """The image containing text is segmented into lines and returns
        a list of the lines

        :return: List of arrays, each array is a line from image

        """
        bin_block_size = int(bin_block_size * 399)
        denoised_img = cv2.fastNlMeansDenoising(binaryimg
                                                (self.image, bin_block_size),
                                                None, 200, 7, 7)
        cropped_image = cropimg(denoised_img)
        plt.imshow(cropped_image)
        plt.show()
        lines = []
        limg = cropped_image.copy()
        last_lin = last_line(limg)
        while not np.sum(limg) == np.sum(last_lin):
            m, n = limg.shape
            p = 0
            for i in range(1, m):
                if np.sum(limg[i, :]) == 0 and\
                   np.sum(limg[i - 1, :]) != 0 and np.sum(limg[i - 5, :]) != 0:
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


def binaryimg(image, bin_block_size):
    """Converts grayscale image to binary image , it gives 1 for black and zero
    for white.

    :param array image: represents the image to be converted to binary

    """
    blur_image = cv2.GaussianBlur(image, (7, 7), 0)
    binary_image = cv2.adaptiveThreshold(blur_image, 1,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         bin_block_size, 15)
    m, n = binary_image.shape
    binary_image = np.asmatrix(binary_image)
    blurred_bin = cv2.GaussianBlur(binary_image, (7, 7), 3)
    return blurred_bin


def cropimg(image):
    """Crops a binary image tightly

    :param array image:

    """
    col_sum = np.where(np.sum(image, axis=0) > 0)
    row_sum = np.where(np.sum(image, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = image[y1:y2 + 1, x1:x2 + 1]
    # import pdb;pdb.set_trace()
    return cropped_image


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
                char_temp = cimg[:, :i]
                h, w = char_temp.shape
                # import pdb;pdb.set_trace()
                # if w < line.shape[1] / 40:
                #     continue
                if np.sum(char_temp) < 0.03 * h * w:
                    p = i
                    continue
                if h > w:
                    pad_t = 0
                    pad_b = 0
                    pad_l = int((h - w) / 2)
                    pad_r = int((h - w) / 2)
                if h < w:
                    pad_r = 0
                    pad_l = 0
                    pad_t = int((w - h) / 2)
                    pad_b = int((w - h) / 2)

                char_temp = c_img.add_padding(char_temp, pad_t,
                                              pad_r, pad_b, pad_l)
                p = i
                chars.append(char_temp)
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
