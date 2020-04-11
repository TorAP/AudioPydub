import numpy as np
import cv2

print (cv2.__version__)
path = "/Users/torarnthpetersen/Desktop/University/4. Semester/AudioPydub/CV/greens.png"
img = cv2.imread(path)
cv2.imshow("hej", img)
cv2.waitKey(1)
