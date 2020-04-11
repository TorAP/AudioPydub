# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wave
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy import signal

# load a trumpet signal
samplingFreq,cleanTrumpetSignal = wave.read('flute-c5-sines.wav')

# cleanTrumpetSignal = cleanTrumpetSignal/2**15 # normalise

samplingFreqChurch,churchImpulse = wave.read('MEDIUM DAMPING ROOM E001 M2S.wav')

# sampling rate 16 kHz
cleanTrumpetSignal = np.reshape(cleanTrumpetSignal, cleanTrumpetSignal.size)
# sampling rate 96 kHz
churchImpulse =np.reshape(churchImpulse, churchImpulse.size)

convoledSignal = np.convolve(churchImpulse,cleanTrumpetSignal,mode='valid')
ipd.Audio(convoledSignal,rate =samplingFreq) 

