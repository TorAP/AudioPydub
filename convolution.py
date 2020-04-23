from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import scipy.signal as sps
import scipy.io.wavfile as tryn
from scipy.io.wavfile import write
import simpleaudio.functionchecks as fc
# AudioPlayer: https://simpleaudio.readthedocs.io/en/latest/simpleaudio.html
import simpleaudio as sa
import wave
from scipy import signal


def convolution (waveFileOne,waveFileTwo):

    samplingFreq1,cleanTrumpetSignal1 = tryn.read(waveFileOne)

    samplingFreq2,cleanTrumpetSignal2 = tryn.read(waveFileTwo)


    ##Use both channels
    # leftSpeakerSignal=cleanTrumpetSignal2[:,0]
    # rightSpeakerSignal=cleanTrumpetSignal2[:,1]

    scaled = np.convolve(cleanTrumpetSignal1,cleanTrumpetSignal2)

    print(cleanTrumpetSignal1,cleanTrumpetSignal2)

    write('test2.wav', 44100, scaled)

    return write

convolution('IR.wav','pluck.wav')


def convert(path):

    # audio_data - object with audio data (must support the buffer interface)
    # num_channels (int) – the number of audio channels
    # bytes_per_sample (int) – the number of bytes per single-channel sample
    # sample_rate (int) – the sample rate in Hz

    wave_read = wave.open(path, 'rb')
    audio_data = wave_read.readframes(wave_read.getnframes())
    num_channels = wave_read.getnchannels()
    bytes_per_sample = wave_read.getsampwidth()
    sample_rate = wave_read.getframerate()

    wave_obj = sa.WaveObject(audio_data, num_channels,
                             bytes_per_sample, sample_rate)

    return wave_obj


def convolution(length1: sa.WaveObject, length2: sa.WaveObject):

    # thefirst = len(length1.audio_data).astype(np.int16)
    # thesecond = len(length2.audio_data).astype(np.int16)

    # thesecond = np.int64(len(length2.audio_data))

    # thesecond = (thesecond.astype(np.int32))

    # thefirst = np.int64(len(length1.audio_data))

    # thefirst = (thefirst.astype(np.int32))

    audio_array = length1.audio_data.astype(np.int16)


    print(length1.audio_data, length2.audio_data)

    final = signal.oaconvolve(thefirst, thesecond)

    num_channels = length2.num_channels
    bytes_per_sample = length2.bytes_per_sample
    sample_rate = length2.sample_rate
    convoluted = sa.WaveObject(
        final, num_channels, bytes_per_sample, sample_rate)

    return convoluted


obj = convert('IR.wav')
obj1 = convert('flute-c5-sines.wav')


# play_obj = obj.play()
# play_obj.wait_done()

# final = np.convolve(len(obj.audio_data),len(obj1.audio_data))

obj2 = convolution(obj, obj1)

play_obj2 = obj2.play()
play_obj2.wait_done()


#audio_array = obj.astype(np.int16)
# fc.LeftRightCheck.run()

# final = np.convolve(len(obj.audio_data),len(obj1.audio_data))

#final = final.astype(np.int16)

# wave_obj1 = sa.WaveObject(final, num_channels, bytes_per_sample, sample_rate)

# wave_obj1.play()

# obj.__new__()
