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

from pathlib import Path
import numpy, scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import librosa, librosa.display
import mir_utils as miru
import seaborn as sns
a=miru.loadAudioArrays()

keys=['kick',
 'rim',
 'tom_high',
 'snare',
 'hihat_open']

def extract_features(signal):
    if len(signal)>1000:
        signal=signal[0:1000]
        l=signal.shape[0]
        zc=librosa.feature.zero_crossing_rate(signal,frame_length=l+1, hop_length=l,)[0,0],
        spec_center=librosa.feature.spectral_centroid(signal,sr=40000,n_fft=l+1, hop_length=l)[0,0],
        zcl=np.log(zc[0])
        scl=spec_center[0]
        ret=[scl,zcl]
        # print(ret)
        return ret
    else:
        return [0,0]

def makefeatDF(k):
    signals=a[k][0:100]
    features=numpy.array([extract_features(x) for x in signals])
    df=pd.DataFrame(features,columns={"log(zero_cross)","centroid"})
    df["type"]=k
    return df


drumDF=pd.concat([makefeatDF(key) for key in keys])
drumDF=drumDF[drumDF["centroid"]!=0]
drumDF.to_csv("zc_and_centroid.csv",index=False)
sns.lmplot(x="log(zero_cross)",y="centroid",hue="type",fit_reg=False,data=drumDF) 

plt.show()