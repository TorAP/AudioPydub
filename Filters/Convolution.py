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
from scipy.signal import sosfiltfilt, butter




from scipy.io.wavfile import _read_riff_chunk
from os.path import getsize

n = -2 # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it
chunk = 1024
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 10
swidth = 2

p = pyaudio.PyAudio()
b,a=signal.iirdesign(0.03,0.07,5,40)
fulldata = np.array([])
print(b)


#samplingFreq1,cleanTrumpetSignal1 = bolge.read('water drip.wav')
#data1 = np.array(cleanTrumpetSignal1)
#data1 = np.fft.rfft(data1)

#print(data1, len(data1))


def callback(in_data, frame_count, time_info, flag):

    samplingFreq1,cleanTrumpetSignal1 = bolge.read('water drip.wav')


    data1 = cleanTrumpetSignal1

    # getting the data from the buffer in in_data
    data = np.frombuffer(in_data, dtype=np.float32)

    data = np.convolve(data1,data)

    print(data)

    # do real fast Fourier transform to get frequency domain
    data = np.fft.rfft(data)

    # data1  = np.fft.rfft(cleanTrumpetSignal1,n=data.size)


    # data = data1*data 

    # inverse transform to get back to time domain
    data = np.fft.irfft(data)

    # convert back to chunks of data
    out_data = np.array(data, dtype=np.float32)
    

    return out_data, pyaudio.paContinue


def callback1(in_data, frame_count, time_info, flag):
    global b,a,fulldata #global variables for filter coefficients and array
    audio_data = np.frombuffer(in_data, dtype=np.int32)

    #filtered in realtime
    #audio_data = signal.filtfilt(b,a,audio_data,padlen=200, irlen=None).astype(np.int32).tostring()
    # audio_data = np.convolve(audio_data,audio_data1)

    sos = butter(4, 0.125, output='sos')
    audio_data = signal.sosfiltfilt(sos,audio_data).astype(np.int32).tostring()

    fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
    return (audio_data, pyaudio.paContinue)

def callback2(in_data, frame_count, time_info, flag):
    audio_data = np.frombuffer(in_data, dtype=np.int32)

    #filtered in fft
    audio_data = np.fft.rfft(audio_data)
    audio_data = audio_data*data1
    audio_data =np.fft.ifft(audio_data)
    # audio_data = np.convolve(audio_data,audio_data1)

    fulldata = np.append(fulldata,audio_data) #saves filtered data in an array
    return (audio_data, pyaudio.paContinue)

def callback3(in_data, frame_count, time_info, flag):
    sos = butter(10, 0.125, output='sos')

    audio_data = np.frombuffer(in_data, dtype=np.int32)

    #filtered in fft
    audio_data = sosfiltfilt(sos,audio_data)
    # audio_data = np.convolve(audio_data,audio_data1)

    return (audio_data, pyaudio.paContinue)

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                frames_per_buffer = 4096,
                stream_callback=None)


stream.start_stream()
print("Recording")



while stream.is_active():
    time.sleep(RECORD_SECONDS)
    stream.stop_stream()
    print("Recording stopped")

stream.close()
p.terminate()
