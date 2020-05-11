#region -- SETUP --------------------------------------------------------------
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

import pyaudio
import sys
import time
import wave

#PyAudio Setup

n = 0  # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it

chunk = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
swidth = 2

effect = False

p = pyaudio.PyAudio()




def combFilter(inAudio, samplingFq, s_delay, decay):
    #Used for the echo

    nData = np.size(inAudio)

    int_offset = int(s_delay * samplingFq)
    outAudio = np.zeros(len(inAudio) + int_offset)

    for i in np.arange(nData):
        if i < len(inAudio) - int_offset:
            outAudio[i + int_offset] = inAudio[i + int_offset] + outAudio[i] * decay
        else:
            outAudio[i + int_offset] = outAudio[i] * decay

    return outAudio


def addVibrato(inputSignal, modDepth, digModFreq, offset=0):
    nData = np.size(inputSignal)
    outputSignal = np.zeros(nData)
    tmpSignal = np.zeros(nData)
    for n in np.arange(nData):
        # calculate delay
        delay = offset + (modDepth/2)*(1-np.cos(digModFreq*n))
        # calculate filter output
        if n < delay:
            outputSignal[n] = 0
        else:
            intDelay = np.int(np.floor(delay))
            tmpSignal[n] = inputSignal[n-intDelay]
            fractionalDelay = delay-intDelay
            apParameter = (1-fractionalDelay)/(1+fractionalDelay)
            outputSignal[n] = apParameter*tmpSignal[n] + \
                tmpSignal[n-1]-apParameter*outputSignal[n-1]
    return outputSignal

def callback(in_data, frame_count, time_info, flag):
    #if effect:
        '''
        # getting the data from the buffer in in_data
        data = np.frombuffer(in_data, dtype=np.float32)

        # do real fast Fourier transform to get frequency domain
        data = np.fft.rfft(data)

        # shifting the array
        data2 = [0]*len(data)
        if n >= 0:
            data2[n:len(data)] = data[0:(len(data)-n)]
            data2[0:n] = data[(len(data)-n):len(data)]
        else:
            data2[0:(len(data)+n)] = data[-n:len(data)]
            data2[(len(data)+n):len(data)] = data[0:-n]
        data = np.array(data2)

        # inverse transform to get back to time domain
        data = np.fft.irfft(data)

        # convert back to chunks of data
        out_data = np.array(data, dtype=np.float32)

        return out_data, pyaudio.paContinue
        '''
        '''
        strength = 4
        s_delay = 2.5
        decay = 0.5
        
        data = np.frombuffer(in_data, dtype=np.float32)

        CF_1 = combFilter(data, RATE, s_delay, decay)
        CF_2 = combFilter(data, RATE, s_delay, decay)
        CF_3 = combFilter(data, RATE, s_delay, decay)
        CF_4 = combFilter(data, RATE, s_delay, decay)

        CF_sample_set = [CF_1, CF_2, CF_3, CF_4]
        CF_sample_maxSize = max([np.size(x) for x in CF_sample_set])

        CF_1.resize(CF_sample_maxSize, refcheck=False)
        CF_2.resize(CF_sample_maxSize, refcheck=False)
        CF_3.resize(CF_sample_maxSize, refcheck=False)
        CF_4.resize(CF_sample_maxSize, refcheck=False)

        prolongedInAudio = np.copy(data)
        prolongedInAudio.resize(CF_sample_maxSize, refcheck=False)
        out_data1 = prolongedInAudio * (1 - strength) + strength * (CF_1 + CF_2 + CF_3 + CF_4)
        out_data2 = np.array(out_data1, dtype=np.float32)
        return out_data2, pyaudio.paContinue
        '''

        print(type(in_data))
        data = np.frombuffer(in_data, dtype=np.float32)
        maxDelay = 0.001*RATE  # samples
        digModFreq = 2*np.pi*2/RATE  # rad/sample
        guitarSignalWithVibrato = addVibrato(data, maxDelay, digModFreq)
        out_data = np.array(guitarSignalWithVibrato, dtype=np.float32)
        print(type(out_data))

        return out_data
        
        '''
        data = np.frombuffer(in_data, dtype=np.float32)
        amp_modA = 0.50
        Hz_modFq = 6*np.pi
        out_data1 = np.copy(data)
        modSignal = 1 - amp_modA * np.cos(Hz_modFq / RATE * np.arange(len(out_data1)))
        print(modSignal)
        out_data1 *= modSignal

        out_data2 = np.array(out_data1, dtype=np.float32)

        return out_data2
        '''
    #else:
    #    return in_data, pyaudio.paContinue


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
res = (4, 11)
square = (width/res[0], height/res[1])
fps, frame_count = 1, 0

minHue, minSat, minVal, maxHue, maxSat, maxVal = 35, 75, 100, 85, 255, 255
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


def changeImage(val):
    cv2.imshow('step', STEP_CODES[val])


def breakHere(trackbar_position):
    breakpoint()


cv2.namedWindow("Calibration")
cv2.resizeWindow("Calibration", 500, 80)
cv2.createTrackbar("Step", "Calibration", 0, 10, changeImage)
cv2.createTrackbar("Break", "Calibration", 0, 1, breakHere)
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
            Rx, Ry, Rw, Rh = cv2.selectROI(
                'Select the object', frame, True, False)
            ROI = frame[Ry:Ry+Rh, Rx:Rx+Rw]
            mask = np.zeros(ROI.shape, np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY, cv2.CV_8UC1)

            #region initial image processing
            frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, cv2.CV_8UC3)
            ROI_HSV = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV, cv2.CV_8UC3)

            lower_threshold = np.array([35, 75, 100])
            upper_threshold = np.array([85, 255, 225])

            frame_thresholded = cv2.inRange(
                ROI_HSV, lower_threshold, upper_threshold)

            frame_dilated = cv2.dilate(
                frame_thresholded, np.ones((25, 25), np.uint8))
            frame_closed = cv2.erode(
                frame_dilated, np.ones((25, 25), np.uint8))
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

            pixels = np.float32(frame_ROI.reshape(-1, 3))

            n_colors = 10
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
            flags = cv2.KMEANS_RANDOM_CENTERS

            _, labels, palette = cv2.kmeans(
                pixels, n_colors, None, criteria, 10, flags)
            _, counts = np.unique(labels, return_counts=True)
            indices = np.argsort(counts)[::-1]

            freqs = np.cumsum(np.hstack([[0], counts[indices]/counts.sum()]))
            rows = np.int_(frame_ROI.shape[0]*freqs)

            dom_patch = np.zeros(shape=frame_ROI.shape, dtype=np.uint8)
            for i in range(len(rows) - 1):
                dom_patch[rows[i]:rows[i + 1], :,
                          :] += np.uint8(palette[indices[i]])

            cv2.imshow("Colors", dom_patch)

            rgbColors = [np.uint8(palette[item]) for item in indices if np.uint8(
                palette[item]).all() != np.array([0, 0, 0]).all()]
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

            lower_threshold = np.array([minHue, minSat, minVal])
            upper_threshold = np.array([maxHue, maxSat, maxVal])

            frame_2_thresholded = cv2.inRange(
                frame_HSV, lower_threshold, upper_threshold)
            frame_2_dilated = cv2.dilate(
                frame_2_thresholded, np.ones((25, 25), np.uint8))
            frame_2_closed = cv2.erode(
                frame_2_dilated, np.ones((25, 25), np.uint8))
            #endregion

            cv2.destroyWindow('Select the object')
            post_calibration = True
            #endregion

        else:
            read_success = True

            #region -- IMAGE PROCESSING ---------------------------------------
            # Convert into HSV color environment
            img_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Define thresholds ([Hue<0; 180>, Saturation<0; 255>, Value<0; 255>])
            lower_threshold = np.array([minHue, minSat, minVal])
            upper_threshold = np.array([maxHue, maxSat, maxVal])

            # Apply threshold; returns a binary image
            img_thresholded = cv2.inRange(
                img_HSV, lower_threshold, upper_threshold)

            # Fiddles
            img_dilated = cv2.dilate(
                img_thresholded, np.ones((25, 25), np.uint8))
            img_closed = cv2.erode(img_dilated, np.ones((25, 25), np.uint8))
            #endregion

            #region -- IMAGE ANALYSIS -----------------------------------------
            try:
                # Center of mass from binary image using scipy; returns y first for some reason
                (My, Mx) = ndimage.measurements.center_of_mass(img_closed)
                posX = Mx//square[0]
                posY = My//square[1]

                print(posX, posY)
                n = -int(posY-5)
                print(n)

                # Create contours from binary image and approximate to reduce noise, only for demonstration purposes tbh
                contours, _ = cv2.findContours(
                    img_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    approx = cv2.approxPolyDP(c, 4, True)
                    cv2.drawContours(frame, [approx], 0, (0, 200, 0), 2)

                # Circle will fail if there's no center of mass and there will be no center of mass if there's no greens
                cv2.circle(img=frame, center=(int(Mx), int(My)),
                           radius=20, color=(255, 255, 0))
                cv2.rectangle(img=frame,
                              pt1=(int(posX*width/res[0]),
                                   int(posY*height/res[1])),
                              pt2=(
                                  int((posX+1)*width/res[0]), int((posY+1)*height/res[1])),
                              color=(255, 255, 0))
                effect = True

            except ValueError:
                print("No green objects found.")
                effect = False
            #endregion

        cv2.namedWindow('Output')
        cv2.resizeWindow('Output', frame_width, frame_height)
        cv2.imshow('Output', frame)

        key = cv2.waitKey(1)

        # If Esc (might not work on Mac?)
        if key == 27:
            break

        # If Bb -> breakpoint
        elif key in (66, 98):
            breakpoint()

        # If Cc -> recalibrate
        elif key in (67, 99):
            post_calibration = False

    frame_count += 1
source_vid.release()
cv2.destroyAllWindows()
stream.stop_stream()
#stream.close()
p.terminate()
