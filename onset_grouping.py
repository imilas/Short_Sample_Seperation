import numpy as np
import pandas as pd
import seaborn as sns
import scipy, matplotlib.pyplot as plt
from pathlib import Path
import mir_utils as miru
import feature_functions as ff
import scipy
import librosa
import madmom

#####
import imp
imp.reload(ff)
#####
a=miru.loadAudioArrays()
kick_signals=a["kicks"]
snare_signals=a["snares"]
clap_signals=a["claps"]


def fitFreq(signals,t="unknown_drum",frameLen=1000,hopLen=4,num_feats=2):
    hopLen=frameLen-1
    def getFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=48000,
            frame_size=frameLen,hop_size=hopLen)
        feat=np.zeros(frameLen)
        for frame in fs[0:4]:
            X=np.absolute(scipy.fft(frame))
            feat+=X
            print(len(X))
        return feat
    onsets=[]
    for s in signals:
        features=getFeat(s)
        onsets.append(features)
    df=pd.DataFrame(onsets)
    feat_cols=[ 'onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df

df=pd.DataFrame()
imp.reload(ff)
for key,signals in a.items():
    print("features from %s"%key)
    # chunk=ff.fitPolyWave(signals)
    chunk=fitFreq(signals,t=key,num_feats=20)
    df=pd.concat([df,chunk])

df=df.fillna(0)
#find most useless features
#df.astype(bool).sum(axis=0).sort_values()

rndperm = np.random.permutation(df.shape[0])
#t-sne
miru.plotTSNE(df,perp=10)