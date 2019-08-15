from __future__ import print_function

import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.io import loadmat
from scipy.fftpack import fft
from scipy.io import wavfile 

import madmom 
import sounddevice as sd

from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
import stanford_mir; stanford_mir.init()

import loading as load
import seaborn as sns
a=load.loadAudioArrays()

kick_signals=a["kicks"]
snare_signals=a["snares"]
clap_signals=a["claps"]

def extract_features(signal):
    return [
        librosa.feature.zero_crossing_rate(signal)[0, 3],
        librosa.feature.spectral_centroid(signal,)[0, 3],
    ]

kick_features = numpy.array([extract_features(x) for x in kick_signals])
snare_features = numpy.array([extract_features(x) for x in snare_signals])
clap_features = numpy.array([extract_features(x) for x in clap_signals])

scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))

kickDF=pd.DataFrame(kick_features,columns={"zero_cross","centroid"})
kickDF["type"]="kick"
snareDF=pd.DataFrame(snare_features,columns={"zero_cross","centroid"})
snareDF["type"]="snare"
clapDF=pd.DataFrame(clap_features,columns={"zero_cross","centroid"})
clapDF["type"]="clap"


drumDF=pd.concat([snareDF,kickDF,clapDF])
sns.lmplot(x="zero_cross",y="centroid",hue="type",fit_reg=False,data=drumDF) 
plt.show()