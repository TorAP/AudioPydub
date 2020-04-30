""" STEP CODES:
    0 = img = original image in BGR color space
    1 = mask = binary image containing the shape of the biggest green object
    2 = img_HSV = original image in HSV color space
    3 = img_thresholded = input image after initial general threshold
    4 = img_eroded = thresholded image after erosion
    5 = img_closed = eroded image after dilation
    6 = img_targetted = input image with the mask applied
    7 = img_ROI = region of interest (original image cropped to only contain the target shape)
    8 = img_2_thresholded = input image after second (automatic) threshold
    9 = img_2_eroded = second-thresholded image after erosion
    10 = img_2_closed = second-eroded image after dilation

    USE THE SLIDER TO SEE THE STEP RESULT
"""

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt


def changeImage(val):
    #cv2.destroyWindow('step')
    cv2.imshow('step', STEP_CODES[val])

cv2.namedWindow("Calibration")
cv2.resizeWindow("Calibration", 500, 40)
cv2.createTrackbar("Step", "Calibration", 0, 10, changeImage)

#region rgb to hsv
def rgb_to_hsv(clr):
    r, g, b = clr[0]/255.0, clr[1]/255.0, clr[2]/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return [int(h/2), int(s*2.55), int(v*2.55)]
#endregion

img = cv2.imread("CV/test.jpg")
mask = np.zeros(img.shape, np.uint8)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#region initial image processing
img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_threshold = np.array([35, 50, 50])
upper_threshold = np.array([85, 255, 200])

img_thresholded = cv2.inRange(img_HSV, lower_threshold, upper_threshold)

img_eroded = cv2.erode(img_thresholded, np.ones((25, 25), np.uint8))
img_closed = cv2.dilate(img_eroded, np.ones((25, 25), np.uint8))
#endregion

#region find dominant colors
contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

biggestShape = max([cv2.contourArea(c) for c in contours])
for c in contours:
    if cv2.contourArea(c) == biggestShape:
        target = c
        break

approx = cv2.approxPolyDP(target, 4, True)
cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)
x, y, w, h = cv2.boundingRect(approx)

img_targetted = cv2.bitwise_and(img, img, mask=mask)
img_ROI = img_targetted[y:y+h, x:x+w]

pixels = np.float32(img_ROI.reshape(-1, 3))

n_colors = 10
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
_, counts = np.unique(labels, return_counts=True)
indices = np.argsort(counts)[::-1]

rgbColors = [np.uint8(palette[item]) for item in indices if np.uint8(
    palette[item]).all() != np.array([0, 0, 0]).all()]
hsvColors = [rgb_to_hsv(item) for item in rgbColors]
#endregion

#region apply threshold
hues = [item[0] for item in hsvColors]
sats = [item[1] for item in hsvColors]
vals = [item[2] for item in hsvColors]

lower_threshold = np.array([min(hues), min(sats), min(vals)])
upper_threshold = np.array([max(hues), max(sats), max(vals)])

img_2_thresholded = cv2.inRange(img_HSV, lower_threshold, upper_threshold)
img_2_eroded = cv2.erode(img_2_thresholded, np.ones((25, 25), np.uint8))
img_2_closed = cv2.dilate(img_2_eroded, np.ones((25, 25), np.uint8))
#endregion

STEP_CODES = [img, mask, img_HSV, img_thresholded, img_eroded, img_closed, img_targetted, img_ROI, img_2_thresholded, img_2_eroded, img_2_closed]

cv2.imshow('step', img)
cv2.waitKey(0)
