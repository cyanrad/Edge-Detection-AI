import Edge_Fuzzy as ef     # the fuzzy logic library
import pixil_manip as pm    # pixel manipulation library
import numpy as np          # for the matrix
import matplotlib.pyplot as plt         # Displaying the plots
import cv2 as cv


def doMatrixOperation(img, simulation):
    returnImg = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    img
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


img = cv.imread("testing2.jpg")
cv.imshow("Testing", img)
cv.waitKey(0)


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
i = 1
j = 1
print()
cv.imshow("Testing", img_gray)

cv.waitKey(0)

bit_fs_array = ef.create_matrixOf_bit_fuzzy_set()
edge_fs = ef.create_edge_fuzzy_set()
control_sim = ef.create_control_sim(bit_fs_array, edge_fs)

test = doMatrixOperation(img_gray, control_sim)
cv.imshow("Testing", test)
cv.waitKey(0)


#testing_mat = np.matrix([[197, 197, 197], [199, 199, 198], [197, 197, 199]])
#testing_mat_dPj = pm.get_dPj_matrix(testing_mat)

#pm.apply_fuzzy_contrast(control_sim, testing_mat_dPj)
# print(ef.isEdge(control_sim))


plt.show()

# waiting for key press for window deletion
cv.destroyAllWindows()
