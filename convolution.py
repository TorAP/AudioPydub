from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as tryn
from scipy.io.wavfile import write
import simpleaudio.functionchecks as fc
#AudioPlayer: https://simpleaudio.readthedocs.io/en/latest/simpleaudio.html
import simpleaudio as sa
import wave



# #def convolution (waveFileOne,waveFileTwo):

#     samplingFreq1,cleanTrumpetSignal1 = tryn.read(waveFileOne)

#     samplingFreq2,cleanTrumpetSignal2 = tryn.read(waveFileTwo)
                                  

#     ##Use both channels
#     leftSpeakerSignal=cleanTrumpetSignal2[:,0]
#     rightSpeakerSignal=cleanTrumpetSignal2[:,1]

#     scaled = np.convolve(cleanTrumpetSignal1,leftSpeakerSignal)

#     write('test.wav', 44100, scaled)
    
#     #return write



def convert(path):


    #audio_data - object with audio data (must support the buffer interface)
    #num_channels (int) – the number of audio channels
    #bytes_per_sample (int) – the number of bytes per single-channel sample
    #sample_rate (int) – the sample rate in Hz

    wave_read = wave.open(path, 'rb')   
    audio_data = wave_read.readframes(wave_read.getnframes())
    num_channels = wave_read.getnchannels()
    bytes_per_sample = wave_read.getsampwidth()
    sample_rate = wave_read.getframerate()

    wave_obj = sa.WaveObject(audio_data, num_channels, bytes_per_sample, sample_rate)

    return wave_obj

def convolution(length1:sa.WaveObject,length2:sa.WaveObject):

    #audio_array = obj.astype(np.int16)
    final = np.convolve(len(length1.audio_data),len(length2.audio_data))
    print(final)

    num_channels = length1.num_channels
    bytes_per_sample = length1.bytes_per_sample
    sample_rate = length1.sample_rate
    convoluted = sa.WaveObject(final, num_channels, bytes_per_sample, sample_rate)

    return convoluted


obj = convert('IR.wav')
obj1 = convert('pluck.wav')



# play_obj = obj.play()
# play_obj.wait_done()

# final = np.convolve(len(obj.audio_data),len(obj1.audio_data))

obj2 = convolution(obj1,obj)

play_obj2 = obj2.play()
play_obj2.wait_done()





#audio_array = obj.astype(np.int16)
#fc.LeftRightCheck.run()

# final = np.convolve(len(obj.audio_data),len(obj1.audio_data))

#final = final.astype(np.int16)

# wave_obj1 = sa.WaveObject(final, num_channels, bytes_per_sample, sample_rate)

# wave_obj1.play()

#obj.__new__()
