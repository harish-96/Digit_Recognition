"""
Center images
~~~~~~~~~~~~~~~~
A module to crop and center images. This is a necessary preprocessing step for
the input data to resemble training data.
"""

import numpy as np
from scipy.ndimage import center_of_mass
from PIL import Image


def add_padding(img, pad_t, pad_r, pad_b, pad_l):
    """Add padding of zeroes to an image.
    Add padding to an array image.

    :param ndarray img: Numpy array of input image which needs to be padded by
        zeros
    :param int pad_t: Number of pixels of paddding on top of the image
    :param int pad_r: Number of pixels of paddding beneath of the image
    :param int pad_b: Number of pixels of paddding to the right of the image
    :param int pad_l: Number of pixels of paddding to the left of the image

    :return: Numpy array of padded image
    """
    height, width = img.shape

    # Adding padding to the left side.
    pad_left = np.zeros((height, pad_l), dtype=np.int)
    img = np.concatenate((pad_left, img), axis=1)

    # Adding padding to the top.
    pad_up = np.zeros((pad_t, pad_l + width))
    img = np.concatenate((pad_up, img), axis=0)

    # Adding padding to the right.
    pad_right = np.zeros((height + pad_t, pad_r))
    img = np.concatenate((img, pad_right), axis=1)

    # Adding padding to the bottom
    pad_bottom = np.zeros((pad_b, pad_l + width + pad_r))
    img = np.concatenate((img, pad_bottom), axis=0)

    return img


# def center_image(img):
#     """Return a centered image.

#     :param ndarray img: Numpy array of input image which needs to be centered

#     :return: Centered image's Numpy array
#     """
#     col_sum = np.where(np.sum(img, axis=0) > 10**-2)
#     row_sum = np.where(np.sum(img, axis=1) > 10**-2)
#     y1, y2 = row_sum[0][0], row_sum[0][-1]
#     x1, x2 = col_sum[0][0], col_sum[0][-1]

#     cropped_image = img[y1:y2 + 1, x1:x2 + 1]
#     # cropped_image = img[y1:y2, x1:x2]
#     plt.imshow(cropped_image)
#     plt.show()
#     zero_axis_fill = (27 - cropped_image.shape[0])
#     one_axis_fill = (27 - cropped_image.shape[1])

#     top = zero_axis_fill / 2 + 1
#     bottom = zero_axis_fill - top + 1
#     left = one_axis_fill / 2 + 1
#     right = one_axis_fill - left + 1
#     padded_image = add_padding(cropped_image, int(top),
#                                int(right), int(bottom), int(left))
#     print(center_of_mass(padded_image))
#     return padded_image


# def center_image(img_in):
#     img = add_padding(img_in, 20, 20, 20, 20)
#     bin_img = (img > (np.mean(img_in) + 10**-2))
#     com = np.rint(center_of_mass(bin_img))
#     x1, x2 = np.abs(com - [0, 13])[1], np.abs(com + [0, 13])[1]
#     y1, y2 = np.abs(com - [13, 0])[0], np.abs(com + [13, 0])[0]

#     cropped_image = img[y1:y2 + 1, x1:x2 + 1]
#     plt.imshow(cropped_image)
#     zero_axis_fill = (27 - cropped_image.shape[0])
#     one_axis_fill = (27 - cropped_image.shape[1])

#     top = zero_axis_fill / 2 + 1
#     bottom = zero_axis_fill - top + 1
#     left = one_axis_fill / 2 + 1
#     right = one_axis_fill - left + 1
#     padded_image = add_padding(cropped_image, int(top),
#                                int(right), int(bottom), int(left))
#     print(center_of_mass(padded_image))

#     return padded_image


def center_image(img):
    """Return a centered image.

    :param ndarray img: Numpy array of input image which needs to be centered

    :return: Centered image's Numpy array
    """
    col_sum = np.where(np.sum(img, axis=0) > 10**-2)
    row_sum = np.where(np.sum(img, axis=1) > 10**-2)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1:y2 + 1, x1:x2 + 1]
    h, w = cropped_image.shape
    if h >= w:
        pad_t = 0
        pad_b = 0
        pad_l = int((h - w) / 2)
        pad_r = int((h - w) / 2)
    if h < w:
        pad_r = 0
        pad_l = 0
        pad_t = int((w - h) / 2)
        pad_b = int((w - h) / 2)

    square_img = add_padding(cropped_image, pad_t,
                             pad_r, pad_b, pad_l)

    image = Image.fromarray(square_img).resize((40, 40))
    centered_img = np.asarray(image.resize((20, 20), Image.ANTIALIAS))
    centered_img = add_padding(centered_img, 4, 4, 4, 4)
    return centered_img
