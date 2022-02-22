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
