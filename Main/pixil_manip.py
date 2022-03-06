import numpy as np
import cv2 as cv

"""
* FILENAME: pixil_manip.py
* DESCRIPTION:
*   contains function for manipulating pixils in the image.
*   WIP
"""


# >> creates matrix with dPj
#
# Parameter: np.matrix of size 3x3
# Return:    np.matrix of size 3x3 with the modified values
#
# Description:
# Will take in a 3x3 matrix representing grayscale pixils
# where each pixil is label Pj(from 1 to 8),
# with the center pixil being P (the main pixil)
# in the return matrix each Pj will be modified as follows:
# dPj = |Pj - P|
#
# | P1, P2, P3 |    | dP1, dP2, dP3 |
# | P4, P,  P5 | -> | dP4,  P,  dP5 |
# | P6, P7, P8 |    | dP6, dP7, dP8 |
#
# P is main pixle and Pj are the surrounding.
# d stands for delta
def get_dPj_matrix(Pj_matrix):
    return_mat = Pj_matrix          # to automatically setup matrix
    P = Pj_matrix[1, 1]             # to store P for operation & final modify

    # calculation range (where we get dPj)
    for i in range(3):
        for j in range(3):
            return_mat[i, j] = abs(P - return_mat[i, j])
    return_mat[1, 1] = P    # reassigning P
    return return_mat

# > sending the contrast to the simulation


def apply_fuzzy_contrast(contrast_simulation, dPj_matrix):
    # getting dPj pixels
    bits = dPj_matrix.flatten()  # flattening the matrix
    bits = np.delete(bits, 4)    # deleting the 5th element(P)
    for i in range(8):
        contrast_simulation.input['contrast' + str(i)] = bits[0, i]


def scale_down(img, max_height=200):
    # max_height is also the new width For simplicity
    # the desired size is 200 rows (height)

    # the resize ratio, what % we will resize the image
    resize_ratio = max_height/img.shape[0]
    if (resize_ratio >= 1):  # if smaller then don't resize
        return img

    new_width = int(img.shape[1] * resize_ratio)

    # doing the resizing
    # in short interpolation is a method for estimating the unkown values between two points
    # ...it much more complex than that though
    # using bilinear interpolation, which is linear interpolation but for two variables (x,y)
    img = cv.resize(img, [new_width, max_height], interpolation=cv.INTER_AREA)
    return img


def scale_up(img, original_shape):
    # if scale down is not applied -> do nothing
    if(img.shape[0] == original_shape[0] and
            img.shape[1] == original_shape[1]):
        return img

    # else scale image back to original size
    img = cv.resize(img, original_shape, interpolation=cv.INTER_AREA)
    return img
