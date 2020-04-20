from pydub import AudioSegment
from pydub.playback import play
import numpy


raw_audio = AudioSegment.from_file("flute-c5-sines.wav", format="wav")

raw_audio1 = AudioSegment.from_file("MEDIUM DAMPING ROOM E001 M2S.wav", format="wav")
                                  

raw_audio.set_sample_width(1)


print(raw_audio1)
play(raw_audio1)