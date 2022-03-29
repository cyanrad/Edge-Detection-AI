# This file is used to take edge detected images
# then turn them into the training data for
# the neural network in the form of a CSV file

# CSV file structure
# in in in in in in in in in out
# P1 P2 P3 P4 P5 P6 P7 P8 P9 isEdge

import cv2 as cv
import sys
import csv


# >> creats the array to be written to the CSV
#
# ARGUMENTS:
# edgeI:     image that is already edge detected
# originalI: image with the contrast pixels (grayscale)
# x(col): pixel x posiiton (in the edgeI img)
# y(row): pixel y position (in the egdeI img)
#
# RETURNS:
# returnArr: np.array
# which contains:_
# P0~P7: from the originalI (where Pn are the surrounding
#                            pixels of the x,y center)
# isEdge: a 1 or 0 value indicating if its an edge
def getReturnArray(edgeI, originalI, x, y):
    returnArr = []

    # >> getting pixels from orignal image
    for i in range(x, x+3):
        for j in range(y, y+3):
            returnArr.append(originalI[i, j])
    # returnArr.pop(4)  # deleting the center element cuz fuck it
    returnArr.pop(4)

    # >> getting edge at <x,y> of edgeI
    if (edgeI[y, x] > 200):  # edge values are 255, but using 200 for error
        returnArr.append(1)
    else:
        returnArr.append(0)

    return returnArr


def show_and_wait(img, window="Testing"):
    cv.imshow(window, img)
    cv.waitKey(0)


def main():

    # we get the image name from the cli args
    if (len(sys.argv) <= 1):
        print("ERROR: edge image file arg not specified")
        sys.exit(2)

    if (len(sys.argv) <= 2):
        print("ERROR: original image file arg not specified")
        sys.exit(2)

    # CSV file init
    f = open('data.csv', 'w', encoding="UTF8", newline="")
    writer = csv.writer(f)

    edgeImage = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)
    originalImage = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    show_and_wait(edgeImage)
    show_and_wait(originalImage)

    for i in range(0, edgeImage.shape[0]):
        print(i)
        for j in range(0, edgeImage.shape[1]):
            returnList = getReturnArray(edgeImage, originalImage, i, j)
            writer.writerow(returnList)
    f.close()


main()
