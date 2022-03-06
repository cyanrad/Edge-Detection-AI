import Edge_Fuzzy as ef     # (custom) the fuzzy logic library
import pixil_manip as pm    # (custom) pixel manipulation library
import numpy as np          # for the matrix, and arrays
import cv2 as cv            # img reading and resizing
# import timeit              # benchmarking
# import matplotlib.pyplot as plt         # Displaying the plots


def apply_contrast(img):
    return_img = img
    intensity_sim = ef.create_intensity_sim()
    for i in range(img.shape[0]):
        print(i)
        for j in range(img.shape[1]):
            intensity_sim.input["intensity"] = return_img[i, j]
            intensity_sim.compute()
            return_img[i, j] = intensity_sim.output['output']
            #print(return_img[i, j])
    return return_img


def detect_edges(img, simulation):
    returnImg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, img.shape[0]-1):
        print(i)
        for j in range(1, img.shape[1]-1):
            # creating matrix
            mat = np.matrix([   # reading gray pixil values
                [img[i-1, j-1], img[i-1, j], img[i-1, j+1]],
                [img[i, j-1], img[i, j], img[i, j+1]],
                [img[i+1, j-1], img[i+1, j], img[i+1, j+1]]])
            mat_dPj = pm.get_dPj_matrix(mat)
            pm.apply_fuzzy_contrast(simulation, mat_dPj)
            if(ef.isEdge(simulation)):
                returnImg[i, j] = (255, 255, 255)
    print("done")
    return returnImg


# for simplicity
def show_and_wait(img, window="Testing"):
    cv.imshow(window, img)
    cv.waitKey(0)


# reading the image and displaying it
img = cv.imread("images/pepper.bmp")
original_shape = img.shape[0:2]  # holding the orignal shape for later
show_and_wait(img)

# converting the image to grayscale, displaying, and compressing it.
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
show_and_wait(img_gray)
#img_gray = pm.scale_down(img_gray, 200)

# >> uncomplete feature, usable but not practical
# img_gray = apply_contrast(img_gray)
# show_and_wait(img_gray)


# creating surrounding bits matrix, edge mf, and the control simulation
bit_fs_array = ef.create_matrixOf_bit_fuzzy_set()
edge_fs = ef.create_edge_fuzzy_set()
control_sim = ef.create_control_sim(bit_fs_array, edge_fs)

# doing the edge detection, and scaling the image back to original size
edge = detect_edges(img_gray, control_sim)
edge = pm.scale_up(edge, original_shape)
show_and_wait(edge)

cv.destroyAllWindows()
