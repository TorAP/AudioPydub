from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as bolge
from scipy.io.wavfile import write
#import simpleaudio.functionchecks as fc
# AudioPlayer: https://simpleaudio.readthedocs.io/en/latest/simpleaudio.html
#import simpleaudio as sa
from scipy import signal
import pyaudio
import sys, time
import wave


from scipy.io.wavfile import _read_riff_chunk
from os.path import getsize

n = -2 # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it
chunk = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
swidth = 2

p = pyaudio.PyAudio()

def callback(in_data, frame_count, time_info, flag):

    samplingFreq1,cleanTrumpetSignal1 = bolge.read('IR.wav')


    data1 = np.frombuffer(cleanTrumpetSignal1, dtype=np.float32)

    # getting the data from the buffer in in_data
    data = np.frombuffer(in_data, dtype=np.float32)

    data = np.convolve(data1,data)

    # do real fast Fourier transform to get frequency domain
    data = np.fft.rfft(data)

    # data1  = np.fft.rfft(cleanTrumpetSignal1,n=data.size)


    # data = data1*data 

    # inverse transform to get back to time domain
    data = np.fft.irfft(data)

    # convert back to chunks of data
    out_data = np.array(data, dtype=np.float32)
    

    return out_data, pyaudio.paContinue


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                stream_callback=callback)

stream.start_stream()
print("Recording")

stream.close()
p.terminate()


