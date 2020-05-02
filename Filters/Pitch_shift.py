from pynput import keyboard

import pyaudio
import sys, time
import numpy as np
import wave

n = 0 # this is how the pitch should change, positive integers increase the frequency, negative integers decrease it
chunk = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
swidth = 2

p = pyaudio.PyAudio()

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


def combFilter(samples, delay_in_mili, decay_factor, sample_rate):
    delay_size = int(delay_in_mili * sample_rate / 1000)
    combFilter_samples = np.zeros(len(samples) + delay_size)

    for i in range(len(samples)):
        if i < len(samples) - delay_size:
            combFilter_samples[i + delay_size] = samples[i + delay_size] + combFilter_samples[i] * decay_factor
        else:
            combFilter_samples[i + delay_size] = combFilter_samples[i] * decay_factor

    return combFilter_samples

def allpassFilter(samples, sample_rate):
    delay_size = int(sample_rate / 10)
    allpassFilter_samples = samples[:]
    decay_factor = 0.131

    # Algorithm:
    for i in range(len(samples)):
        if (i - delay_size >= 0):
            allpassFilter_samples[i] += -decay_factor * allpassFilter_samples[i - delay_size]
        if (i - delay_size >= 1):
            allpassFilter_samples[i] += decay_factor * allpassFilter_samples[i + 20 - delay_size]

    max_val = np.amax([abs(item) for item in allpassFilter_samples])
    allpassFilter_samples = [item/max_val for item in allpassFilter_samples]

    return allpassFilter_samples


defaults = {
    "delay": 500,
    "decay": 0.5,
    "moist": 1}

def callback1(in_data, frame_count, time_info, flag):
    # getting the data from the buffer in in_data
    data = np.frombuffer(in_data, dtype=float)

    CF_samples_1 = combFilter(data, defaults["delay"], defaults["decay"], RATE)
    CF_samples_2 = combFilter(data, defaults["delay"] - 11.73, defaults["decay"] - 0.1313, RATE)
    CF_samples_3 = combFilter(data, defaults["delay"] + 19.31, defaults["decay"] - 0.2743, RATE)
    CF_samples_4 = combFilter(data, defaults["delay"] - 7.97, defaults["decay"] - 0.31, RATE)

    CF_sample_set = [CF_samples_1, CF_samples_2, CF_samples_3, CF_samples_4]
    CF_sample_maxSize = max([np.size(x) for x in CF_sample_set])

    CF_samples_1.resize(CF_sample_maxSize, refcheck=False)
    CF_samples_2.resize(CF_sample_maxSize, refcheck=False)
    CF_samples_3.resize(CF_sample_maxSize, refcheck=False)
    CF_samples_4.resize(CF_sample_maxSize, refcheck=False)

    postcomb_data = CF_samples_1 + CF_samples_2 + CF_samples_3 + CF_samples_4

    moist_mix = [0] * len(data)

    for i in range(len(data)):
        moist_mix[i] = ((1 - defaults["moist"]) * data[i]) + (defaults["moist"] * postcomb_data[i])

    reverbed_samples = np.array(moist_mix)

    # convert back to chunks of data
    out_data = np.array(reverbed_samples, dtype=np.float32)

    return out_data, pyaudio.paContinue


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                output=True,
                input=True,
                stream_callback=callback)


stream.start_stream()
print("Recording")

def on_press(key):
    """
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
    """
    if key == '1':
        n = n - 1
    if key == '2':
        n = n + 1


def on_release(key):
    print('{0} released'.format(
        key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False


# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()


while stream.is_active():
    on_press(key)
    time.sleep(RECORD_SECONDS)
    stream.stop_stream()
    print("Recording stopped")

stream.close()
p.terminate()
