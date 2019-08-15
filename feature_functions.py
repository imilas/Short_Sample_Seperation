import madmom 
import numpy as np
import seaborn as sns
import pandas as pd
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import librosa, librosa.display

def getOnsetDF(signals,t="u"):
    def getOnsets(x):
        spec = madmom.audio.spectrogram.Spectrogram(x, frame_size=300, hop_size=10)
        X=madmom.features.onsets.high_frequency_content(spec)
        return X 
    onsets=[]
    for s in signals:
        onset=getOnsets(s)
        onsets.append(onset[0:300])
    
    df=pd.DataFrame(onsets)
    feat_cols=['onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df

def fitPolyWave(signals,t="unknown_drum",frameLen=20,hopLen=19,polyDeg=1,num_feats=10000):
    def getFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=48000,
            frame_size=frameLen,hop_size=hopLen)
        feats=[]
        for frame in fs[0:num_feats]:
            try:
                feat=np.polyfit(frame,np.linspace(0,1,frame.shape[0]),deg=polyDeg)
                feats.extend(feat)
            except:
                print("bad frame")
                continue
        return feats[0:num_feats] 
    analyzed=[]
    for s in signals:
        features=getFeat(s)
        analyzed.append(features)
    
    df=pd.DataFrame(analyzed)
    feat_cols=[ 'onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df



