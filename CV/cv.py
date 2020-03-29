import numpy as np
import cv2
from scipy import ndimage

source_vid = cv2.VideoCapture('input.mp4')

while True:
    _, frame = source_vid.read()

    img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([35, 50, 50])
    upper_threshold = np.array([85, 255, 255])

    img_thresholded = cv2.inRange(img_HSV, lower_threshold, upper_threshold)

    img_eroded = cv2.erode(img_thresholded, np.ones((25, 25), np.uint8))
    img_closed = cv2.dilate(img_eroded, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        approx = cv2.approxPolyDP(c, 4, True)
        cv2.drawContours(frame, [approx], 0, (0, 200, 0), 2)
        #center_of_mass = ndimage.measurements.center_of_mass(approx)
        #cv2.circle(img=frame, center=center_of_mass, radius=20, color=(200, 0, 0))

    cv2.imshow('res', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

source_vid.release()
cv2.destroyAllWindows()
