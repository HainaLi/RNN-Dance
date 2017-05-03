__author__ = 'Leandra'

from pyAudioAnalysis import audioAnalysis
import numpy as np





if __name__ == "__main__":
    stFeatures = audioAnalysis.beatExtractionWrapper("wavs/24k_magic.wav", True)
    energy_x = np.zeros(len(stFeatures[:][0]))
    print(len(stFeatures[:][0]))
    energy_y = np.zeros(len(stFeatures[:][0]))
    for i in range(len(stFeatures[:][0])):
        energy_y[i] = stFeatures[1][i]
        energy_x[i] = (i+1)*50