import scipy.io.wavfile as wave
import pydub 
from pydub import AudioSegment
from pydub.playback import play
import matplotlib as plt
import numpy as np

final = AudioSegment.from_wav("beat.wav")
loop = AudioSegment.from_wav("metallic-drums.wav")

final