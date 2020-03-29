import numpy as np
import cv2

source_image = cv2.imread('greens.png')

img_HSV = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV)
lower_threshold = np.array([35, 0, 0])
upper_threshold = np.array([85, 255, 255])

img_thresholded = cv2.inRange(img_HSV, lower_threshold, upper_threshold)

cv2.imwrite('alpha.png', img_thresholded)

#cv2.imshow('res', img_thresholded)

cv2.waitKey(0)
cv2.destroyAllWindows()
