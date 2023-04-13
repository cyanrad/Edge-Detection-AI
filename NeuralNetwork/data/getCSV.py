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
# x(col): pixel x position (in the edgeI img)
# y(row): pixel y position (in the edgeI img)
#
# RETURNS:
# returnArr: np.array
# which contains:_
# P0~P7: from the originalI (where Pn are the surrounding
#                            pixels of the x,y center)
# isEdge: a 1 or 0 value indicating if its an edge
def getReturnArray(edgeI, originalI, x, y):
    returnArr = []                   # the array we will return
    centerVal = originalI[x+1, y+1]  # temporarly holding the center pixel

    # >> getting pixels from orignal image
    for i in range(x, x+3):
        for j in range(y, y+3):
            # getting the contrast from pixel<i,j>
            returnArr.append(abs(centerVal-originalI[i, j])/255)
    # deleting the center element cuz we don't need it (always 0)
    returnArr.pop(4)

    # >> getting edge at <x,y> of edgeI
    if (edgeI[x, y] > 200):  # edge values are 255, but using 200 for error
        returnArr.append(1)
    else:
        returnArr.append(0)

    return returnArr


# >> to display the image
def show_and_wait(img, window="Testing"):
    cv.imshow(window, img)
    cv.waitKey(0)


def main():
    # >> Getting images file location
    # the Original image
    if (len(sys.argv) <= 1):
        print("ERROR: original image file arg not specified")
        sys.exit(2)

    # the Edge image
    if (len(sys.argv) <= 2):
        print("ERROR: edge image file arg not specified")
        sys.exit(2)

    # >> CSV file & writer init
    f = open('hand_written_norm.csv', 'w', encoding="UTF8", newline="")
    writer = csv.writer(f)

    # >> reading the images
    edgeImage = cv.imread(sys.argv[2], cv.IMREAD_GRAYSCALE)
    originalImage = cv.imread(sys.argv[1], cv.IMREAD_GRAYSCALE)
    show_and_wait(edgeImage)
    show_and_wait(originalImage)

    # >> looping over image and writing data
    count1 = 0  # the number for edges the system detected
    for i in range(0, edgeImage.shape[0]):
        print(i)    # as a counter for progress
        for j in range(0, edgeImage.shape[1]):
            # getting the data for index i=y,j=x
            returnList = getReturnArray(edgeImage, originalImage, i, j)

            # for each edge we write a non-edge so they are balanced
            # if the number of non-edge is significantly greater then data is useless
            if (returnList[8] == 0):    # non-edge
                # this is for the sake of keeping the data 50/50 in edge to non-e ratio
                if (count1 > 0):        # if edge count is 0, we don't write a non-edge
                    writer.writerow(returnList)
                    count1 = count1-1
            elif(returnList[8] == 1):   # edge
                writer.writerow(returnList)
                count1 = count1+1

    f.close()  # closing the CSV file


main()
