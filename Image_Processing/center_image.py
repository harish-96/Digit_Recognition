import numpy as np


def add_padding(img, pad_t, pad_r, pad_b, pad_l):
    """Add padding of zeroes to an image.
    Add padding to an array image.
    :param img:
    :param pad_t:
    :param pad_r:
    :param pad_b:
    :param pad_l:
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


def center_image(img):
    """Return a centered image.
    :param img:
    """
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    cropped_image = img[y1 - 1:y2 + 1, x1 - 1:x2 + 1]
    zero_axis_fill = (27 - cropped_image.shape[0])
    one_axis_fill = (27 - cropped_image.shape[1])

    top = zero_axis_fill / 2 + 1
    bottom = zero_axis_fill - top + 1
    left = one_axis_fill / 2 + 1
    right = one_axis_fill - left + 1
    padded_image = add_padding(cropped_image, int(top),
                               int(right), int(bottom), int(left))

    return padded_image
