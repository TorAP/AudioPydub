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





def convolution (waveFileOne,waveFileTwo):

    samplingFreq1,cleanTrumpetSignal1 = bolge.read(waveFileOne)

    samplingFreq2,cleanTrumpetSignal2 = bolge.read(waveFileTwo)


    ##Use both channels and convert to mono
    
    leftSpeakerSignal=cleanTrumpetSignal2[:,0]
    rightSpeakerSignal=cleanTrumpetSignal2[:,1]

    cleanTrumpetSignal2 =leftSpeakerSignal+rightSpeakerSignal 


    scaled = signal.oaconvolve(cleanTrumpetSignal1,cleanTrumpetSignal2)

    # print(scaled,cleanTrumpetSignal1,cleanTrumpetSignal2)

    #Save file as test4
    write('test4.wav', 44100, scaled)

    return write

convolution('pluck.wav','MEDIUM DAMPING ROOM E001 M2S.wav')


def callback(in_data, frame_count, time_info, flag):
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

p = pyaudio.PyAudio()


n = -2 # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it
chunk = 1024
FORMAT = pyaudio.paInt16
channels = 1
RATE = 44100
RECORD_SECONDS = 5
seconds =3
swidth = 2

filename = "NewOutput.wav"


print("Recording")
stream = p.open(format=FORMAT,
                channels=channels,
                rate=RATE,
                frames_per_buffer=chunk,
                input=True,)
                #stream_callback=callback)


frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds

for i in range(0, int(RATE / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)


# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')
 
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()