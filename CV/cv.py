#region -- SETUP --------------------------------------------------------------
import numpy as np
import cv2
from scipy import ndimage

source_vid = cv2.VideoCapture(0) #Auto-closes the whole thing
read_success = False
post_calibration = False

height = source_vid.get(4)
width = source_vid.get(3)
res = (32, 18)
square = (width/res[0], height/res[1])
#endregion

#region -- CALIBRATION WINDOW -------------------------------------------------
def nothing():
    pass

cv2.namedWindow("Calibration")
cv2.resizeWindow("Calibration", 500, 600)
cv2.createTrackbar("L-H", "Calibration", 105, 360, nothing)
cv2.createTrackbar("L-S", "Calibration", 20, 100, nothing)
cv2.createTrackbar("L-V", "Calibration", 8, 100, nothing)
cv2.createTrackbar("U-H", "Calibration", 185, 360, nothing)
cv2.createTrackbar("U-S", "Calibration", 75, 100, nothing)
cv2.createTrackbar("U-V", "Calibration", 60, 100, nothing)
cv2.createTrackbar("Dilate", "Calibration", 19, 50, nothing)
cv2.createTrackbar("Erode", "Calibration", 3, 50, nothing)
cv2.createTrackbar("Open/Close", "Calibration", 1, 1, nothing)
#endregion

while True:
    _, frame = source_vid.read()

    # VIDEO FEED DID NOT START (SOURCE NOT FOUND)
    if frame is None and read_success == False:
        raise Exception("Source not found. Webcam input will be 0 or 1, for local files try full path to file if everything else fails.")
    
    # VIDEO FEED ENDED
    elif frame is None and read_success == True:
        source_vid.release()
        cv2.destroyAllWindows()

    # IN VIDEO FEED
    else:
        if post_calibration:
            read_success = True

            #region -- IMAGE PROCESSING ---------------------------------------
            # Convert into HSV color environment
            img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define thresholds ([Hue<0; 180>, Saturation<0; 255>, Value<0; 255>])
            lower_threshold = np.array([35, 50, 50])
            upper_threshold = np.array([85, 255, 200])

            # Apply threshold; returns a binary image
            img_thresholded = cv2.inRange(img_HSV, lower_threshold, upper_threshold)

            # Fiddles
            img_eroded = cv2.erode(img_thresholded, np.ones((25, 25), np.uint8))
            img_closed = cv2.dilate(img_eroded, np.ones((25, 25), np.uint8))
            #endregion

            #region -- IMAGE ANALYSIS -----------------------------------------
            # Center of mass from binary image using scipy; returns y first for some reason
            (My, Mx) = ndimage.measurements.center_of_mass(img_closed)
            posX = Mx//square[0]
            posY = My//square[1]

            # Create contours from binary image and approximate to reduce noise, only for demonstration purposes tbh
            contours, _ = cv2.findContours(img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for c in contours:
                approx = cv2.approxPolyDP(c, 4, True)
                cv2.drawContours(frame, [approx], 0, (0, 200, 0), 2)

            # Circle will fail if there's no center of mass and there will be no center of mass if there's no greens
            try: cv2.circle(img=frame, center=(int(Mx), int(My)), radius=20, color=(255, 255, 0))
            except ValueError: pass

            cv2.rectangle(img=frame,
                          pt1=(int(posX*width/res[0]), int(posY*height/res[1])),
                          pt2=(int((posX+1)*width/res[0]), int((posY+1)*height/res[1])),
                          color=(255, 255, 0))

        elif not post_calibration:
            pass

        cv2.imshow('res', frame)

        # If Esc (might not work on Mac?)
        key = cv2.waitKey(1)
    
        if key == 27:
            break

source_vid.release()
cv2.destroyAllWindows()
