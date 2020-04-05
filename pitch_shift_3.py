import numpy
import time
from aupyom import Sound, Sampler

audio_file = "/Users/mcsep/Documents/Git Projects/P4/AudioPydub-master/AudioPydub/in.wav"

s1 = Sound.from_file(audio_file)

sampler = Sampler()

sampler.play(s1)

while s1.playing:
    s1.pitch_shift = -10
