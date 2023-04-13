import cv2 as cv
import timeit

# getting & showing image
img = cv.imread('cir_noise_gussian_0.45.png')
cv.imshow("Testing", img)
cv.waitKey(0)


# Grayscale conversion
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
cv.imshow("Testing", img_blur)
cv.waitKey(0)

# Sobel Edge detection
img_edge1 = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
img_edge2 = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
img_edge3 = cv.Sobel(src=img_gray, ddepth=cv.CV_64F, dx=1, dy=1, ksize=3)
cv.imshow("Testing", img_edge1)
cv.waitKey(0)

start_time = timeit.default_timer()
edges = cv.Canny(image=img_blur, threshold1=100,
                 threshold2=200)  # Canny Edge Detection
print(timeit.default_timer() - start_time)
cv.imshow('Testing', edges)
cv.waitKey(0)

cv.imwrite("cir_noise_gussian_0.45_edge.png", edges)


# waiting for key press for window deletion
cv.destroyAllWindows()
