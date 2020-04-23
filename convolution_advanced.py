
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
