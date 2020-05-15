#region -- SETUP --------------------------------------------------------------
import cv2
from numpy import size, zeros, arange, cos, floor, frombuffer, float32, fft, ones, array, uint8, unique, argsort, cumsum, hstack, pi
from scipy import ndimage
from pyaudio import PyAudio, paFloat32, paContinue

#PyAudio Setup

n = 0  # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it
m = 0
chunk = 1024
FORMAT = paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
swidth = 2

effect = False

p = PyAudio()


def combFilter(inAudio, samplingFq, s_delay, decay):
    #Used for the echo

    nData = size(inAudio)

    int_offset = int(s_delay * samplingFq)
    outAudio = zeros(len(inAudio) + int_offset)

    for i in arange(nData):
        if i < len(inAudio) - int_offset:
            outAudio[i + int_offset] = inAudio[i + int_offset] + outAudio[i] * decay
        else:
            outAudio[i + int_offset] = outAudio[i] * decay

    return outAudio


def vibrato(inAudio, samplingFq, ms_strength, Hz_modFq, ms_offset=0):
    outAudio = zeros(len(inAudio))
    allPassAudio = zeros(len(inAudio))

    for i in range(len(outAudio)):
        ms_delay = (ms_strength / 2) * (1 - cos(Hz_modFq * i))

        if i > ms_delay:
            ms_intDelay = int(floor(ms_delay))
            allPassAudio[i] = inAudio[i - ms_intDelay]
            ms_fractionalDelay = ms_delay - ms_intDelay
            b = (1 - ms_fractionalDelay) / (1 + ms_fractionalDelay)

            outAudio[i] = b * allPassAudio[i] + allPassAudio[i-1] - b * outAudio[i-1]

    return outAudio


def callback(in_data, frame_count, time_info, flag):
    #if effect:

        if m == 2:
            # getting the data from the buffer in in_data
            data = frombuffer(in_data, dtype=float32)
            #print(type(in_data))
            # do real fast Fourier transform to get frequency domain
            data = fft.rfft(data)

            # shifting the array
            data2 = [0]*len(data)
            if n >= 0:
                data2[n:len(data)] = data[0:(len(data)-n)]
                data2[0:n] = data[(len(data)-n):len(data)]
            else:
                data2[0:(len(data)+n)] = data[-n:len(data)]
                data2[(len(data)+n):len(data)] = data[0:-n]
            data = array(data2)

            # inverse transform to get back to time domain
            data = fft.irfft(data)

            # convert back to chunks of data
            out_data = array(data, dtype=float32)
            #print(type(out_data))
            return out_data, paContinue
        
        elif m == 0:
            
            data = frombuffer(in_data, dtype=float32)

            ms_strength = 15 / 1000 * RATE

            Hz_modFq = 10 * pi / RATE

            out_data1 = vibrato(inAudio=data, samplingFq=RATE, ms_strength=ms_strength, Hz_modFq=Hz_modFq)

            out_data2 = array(out_data1, dtype=float32)

            return out_data2, paContinue

        else:
            data = frombuffer(in_data, dtype=float32)
            return data, paContinue


        
    #else:
    #    return in_data, paContinue


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                stream_callback=callback)

stream.start_stream()

#CV Setup

frame_width = 1280
frame_height = 720

source_vid = cv2.VideoCapture(0)  # Auto-closes the whole thing
source_vid.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
source_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

read_success = False
post_calibration = False

height = source_vid.get(4)
width = source_vid.get(3)
res = (3, 11)
square = (width/res[0], height/res[1])
fps, frame_count = 1, 0

minHue, minSat, minVal, maxHue, maxSat, maxVal = 35, 75, 100, 95, 255, 255

UI_x = 440
UI_y = 620
#endregion

#region -- CALIBRATION WINDOW -------------------------------------------------


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

while True:
    _, frame = source_vid.read()

    # VIDEO FEED DID NOT START (SOURCE NOT FOUND)
    if frame is None and read_success == False:
        raise Exception(
            "Source not found. Webcam input will be 0 or 1, for local files try full path to file if everything else fails.")

    # VIDEO FEED ENDED
    elif frame is None and read_success == True:
        source_vid.release()
        cv2.destroyAllWindows()

    # IN VIDEO FEED
    elif frame_count % fps == 0:
        frame = cv2.flip(frame, 1)
        if not post_calibration:
            #region -- CALIBRATION --------------------------------------------
            calibInfo = cv2.imread("CV/src/calib.png")
            cv2.imshow("Mark the object", calibInfo)
            cv2.waitKey()

            try:
                Rx, Ry, Rw, Rh = cv2.selectROI(
                    'Mark the object', frame, True, False)
                ROI = frame[Ry:Ry+Rh, Rx:Rx+Rw]
                mask = zeros(ROI.shape, uint8)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY, cv2.CV_8UC1)

                #region initial image processing
                frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, cv2.CV_8UC3)
                ROI_HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV, cv2.CV_8UC3)

                lower_threshold = array([35, 75, 100])
                upper_threshold = array([95, 255, 225])

                frame_thresholded = cv2.inRange(
                    ROI_HSV, lower_threshold, upper_threshold)

                frame_dilated = cv2.dilate(
                    frame_thresholded, ones((25, 25), uint8))
                frame_closed = cv2.erode(
                    frame_dilated, ones((25, 25), uint8))
                #endregion

                #region find dominant colors
                contours, _ = cv2.findContours(
                    frame_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # If there's no contours, continue to the next while iteration without setting post_calibration to True
                try:
                    biggestShape = max([cv2.contourArea(c) for c in contours])
                except ValueError:
                    print("No green objects found, calibration prompt will run again.")
                    continue

                for c in contours:
                    if cv2.contourArea(c) == biggestShape:
                        target = c
                        break

                approx = cv2.approxPolyDP(target, 4, True)
                cv2.drawContours(mask, [approx], 0, (255, 255, 255), -1)
                x, y, w, h = cv2.boundingRect(approx)

                frame_targetted = cv2.bitwise_and(ROI, ROI, mask=mask)
                frame_ROI = frame_targetted[y:y+h, x:x+w]

                pixels = float32(frame_ROI.reshape(-1, 3))

                n_colors = 10
                criteria = (cv2.TERM_CRITERIA_EPS +
                            cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
                flags = cv2.KMEANS_RANDOM_CENTERS

                _, labels, palette = cv2.kmeans(
                    pixels, n_colors, None, criteria, 10, flags)
                _, counts = unique(labels, return_counts=True)
                indices = argsort(counts)[::-1]

                """freqs = cumsum(hstack([[0], counts[indices]/counts.sum()]))
                rows = int_(frame_ROI.shape[0]*freqs)

                dom_patch = zeros(shape=frame_ROI.shape, dtype=uint8)
                for i in range(len(rows) - 1):
                    dom_patch[rows[i]:rows[i + 1], :,
                            :] += uint8(palette[indices[i]])

                cv2.imshow("Colors", dom_patch)"""

                rgbColors = [uint8(palette[item]) for item in indices if uint8(
                    palette[item]).all() != array([0, 0, 0]).all()]
                hsvColors = [rgb_to_hsv(item) for item in rgbColors]
                #endregion

                #region apply threshold
                hues = [item[0] for item in hsvColors]
                sats = [item[1] for item in hsvColors]
                vals = [item[2] for item in hsvColors]

                tolerance = 0
                minHue = min(hues) - tolerance if min(hues) - \
                    tolerance >= 35 else 35
                minSat = min(sats) - tolerance if min(sats) - \
                    tolerance >= 75 else 75
                minVal = min(vals) - tolerance if min(vals) - \
                    tolerance >= 100 else 100
                maxHue = max(hues) + tolerance if max(hues) + \
                    tolerance <= 85 else 85
                maxSat = max(sats) + tolerance if max(sats) + \
                    tolerance <= 255 else 255
                maxVal = max(vals) + tolerance if max(vals) + \
                    tolerance <= 255 else 255

                lower_threshold = array([minHue, minSat, minVal])
                upper_threshold = array([maxHue, maxSat, maxVal])

                frame_2_thresholded = cv2.inRange(
                    frame_HSV, lower_threshold, upper_threshold)
                frame_2_dilated = cv2.dilate(
                    frame_2_thresholded, ones((25, 25), uint8))
                frame_2_closed = cv2.erode(
                    frame_2_dilated, ones((25, 25), uint8))
                #endregion

            except Exception as e:
                print("Calibration failed, set to default values.")
                minHue, minSat, minVal, maxHue, maxSat, maxVal = 35, 75, 100, 85, 255, 255
            
            finally:
                cv2.destroyWindow('Mark the object')
                post_calibration = True
            #endregion

        else:
            read_success = True

            #region -- IMAGE PROCESSING ---------------------------------------
            # Convert into HSV color environment
            img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define thresholds ([Hue<0; 180>, Saturation<0; 255>, Value<0; 255>])
            lower_threshold = array([minHue, minSat, minVal])
            upper_threshold = array([maxHue, maxSat, maxVal])

            # Apply threshold; returns a binary image
            img_thresholded = cv2.inRange(
                img_HSV, lower_threshold, upper_threshold)

            # Fiddles
            img_dilated = cv2.dilate(
                img_thresholded, ones((25, 25), uint8))
            img_closed = cv2.erode(img_dilated, ones((25, 25), uint8))
            #endregion

            #region -- IMAGE ANALYSIS -----------------------------------------
            try:
                # Center of mass from binary image using scipy; returns y first for some reason
                (My, Mx) = ndimage.measurements.center_of_mass(img_closed)
                posX = Mx//square[0]
                posY = My//square[1]

                #print(posX, posY)
                n = -int(posY-5)
                m = int(posX)
                
                # Create contours from binary image and approximate to reduce noise, only for demonstration purposes tbh
                contours, _ = cv2.findContours(
                    img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    approx = cv2.approxPolyDP(c, 4, True)
                    cv2.drawContours(frame, [approx], 0, (0, 200, 0), 2)

                # Circle will fail if there's no center of mass and there will be no center of mass if there's no greens
                cv2.circle(img=frame, center=(int(Mx), int(My)),
                           radius=20, color=(255, 255, 0))
                effect = True

                if m == 2:
                    UI_source = "CV/src/" + str(n) + ".png"
                    
                    cv2.rectangle(img=frame,
                                  pt1=(int(posX*width/res[0]),
                                       int(posY*height/res[1])),
                                  pt2=(int((posX+1)*width/res[0]),
                                       int((posY+1)*height/res[1])),
                                  color=(248, 248, 248))
                
                else:
                    cv2.rectangle(img=frame,
                                  pt1=(int(posX*width/res[0]), 0),
                                  pt2=(int((posX+1)*width/res[0]), frame_height),
                                  color=(248, 248, 248))

                    if m == 1:
                        UI_source = "CV/src/no.png"

                    else:
                        UI_source = "CV/src/alien.png"

                UI = cv2.imread(UI_source)
                frame[UI_y : UI_y+UI.shape[0],
                      UI_x : UI_x+UI.shape[1]] = UI

            except ValueError:
                UI = cv2.imread("CV/src/warning.png")
                frame[UI_y - 37 : UI_y + UI.shape[0] - 37,
                      UI_x : UI_x + UI.shape[1]] = UI
                effect = False
            #endregion

        cv2.namedWindow('Output')
        cv2.resizeWindow('Output', frame_width, frame_height)
        cv2.imshow('Output', frame)

        key = cv2.waitKey(1)

        # If Esc
        if key == 27:
            break

        # If Bb -> breakpoint
        elif key in (66, 98):
            breakpoint()

        # If Cc -> recalibrate
        elif key in (67, 99):
            post_calibration = False

        # If Dd -> defaults
        elif key in (68, 100):
            minHue, minSat, minVal, maxHue, maxSat, maxVal = 35, 75, 100, 85, 255, 255

    frame_count += 1
source_vid.release()
cv2.destroyAllWindows()
stream.stop_stream()
#stream.close()
p.terminate()
