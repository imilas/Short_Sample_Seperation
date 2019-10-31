import numpy as np
import pandas as pd
import seaborn as sns
import scipy, matplotlib.pyplot as plt
from pathlib import Path
import mir_utils as miru
import feature_functions as ff
import scipy
import librosa,librosa.display
import madmom
import pandas as pd
import multiprocessing

df=miru.audioFrames(load=True)

def fitFreq(rows,frameLen=100,hopLen=100,numFrames=100):
    rows.reset_index()
    hopLen=frameLen-1
    def getFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=40000,
            frame_size=frameLen,hop_size=hopLen)
        feat=np.zeros(frameLen)
        for frame in fs:
            X=np.absolute(scipy.fft(frame))
            feat+=X
        return feat
    onsets=[]
    for i,s in rows.iterrows():
        features=getFeat(s["audio"])
        onsets.append(features)
    df=pd.DataFrame(onsets)
    feat_cols=[ 'onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=rows.reset_index().label
    df["path"]=rows.reset_index().path

    return df

def getFeats(d):
    d_copy=d.copy()
    ffd=fitFreq(d_copy)
    return ffd
    
num_processes = multiprocessing.cpu_count()
chunks = np.array_split(df,num_processes)
pool = multiprocessing.Pool(processes=num_processes)
result = pool.map(getFeats, chunks)
feats=pd.concat(result)

print(feats)

rndperm = np.random.permutation(feats.shape[0])
feats.to_csv("feat_frequency_bins.csv",index=False)