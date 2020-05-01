import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import IPython.display as ipd

def allpassReverberator(inputSignal, delay, apParameter):
    nData = np.size(inputSignal)
    outputSignal = np.zeros(nData)
    for n in np.arange(nData):
        if n < delay:
            outputSignal[n] = inputSignal[n]
        else:
            outputSignal[n] = apParameter*inputSignal[n] + inputSignal[n-delay] - \
                apParameter*outputSignal[n-delay]
    return outputSignal

def shroederReverb(inputSignal, mixingParams, plainDelays, plainGains, allpassDelays, apParams):
    nData = np.size(inputSignal)
    tmpSignal = np.zeros(nData)
    # run the plain reverberators in parallel
    nPlainReverberators = np.size(plainDelays)
    for ii in np.arange(nPlainReverberators):
        tmpSignal = tmpSignal + \
            mixingParams[ii]*plainReverberator(inputSignal, plainDelays[ii], plainGains[ii])
    # run the allpass reverberators in series
    nAllpassReverberators = np.size(allpassDelays)
    for ii in np.arange(nAllpassReverberators):
        tmpSignal = allpassReverberator(tmpSignal, allpassDelays[ii], apParams[ii])
    return tmpSignal

mixingParams = np.array([0.3, 0.25, 0.25, 0,20])
plainDelays = np.array([1553, 1613, 1493, 1153])
allpassDelays = np.array([223, 443])
apParams = np.array([-0.7, -0.7])
reverbTime = 0.8 # seconds
plainGains = plainGainFromReverbTime(reverbTime, plainDelays, samplingFreq)
# compute the impulse response of the room
irLength = np.int(np.floor(reverbTime*samplingFreq))
impulse = np.r_[np.array([1]),np.zeros(irLength-1)]
impulseResponse = guitarSignalWithReverb = \
    shroederReverb(impulse, mixingParams, plainDelays, plainGains, allpassDelays, apParams)

guitarSignalWithReverb = \
shroederReverb(guitarSignal, mixingParams, plainDelays, plainGains, allpassDelays, apParams)

