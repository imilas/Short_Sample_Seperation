import madmom 
import numpy as np
import seaborn as sns
import pandas as pd
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import librosa, librosa.display
sr=40000
def fitMels(signals,t="unknown_drum",num_feats=2):
    def getFeat(x):
        X = librosa.feature.melspectrogram(S=x,n_mels=num_feats, sr=sr,)
        return X
    feats=[]
    for s in signals:
        features=getFeat(s)
        feats.append(features)
    df=pd.DataFrame(feats)
    feat_cols=[ 'feat'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df

def fitFreq(signals,t="unknown_drum",frameLen=100,hopLen=100,numFrames=100):
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
    for s in signals:
        features=getFeat(s)
        onsets.append(features)
    df=pd.DataFrame(onsets)
    feat_cols=[ 'onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df
    
def getOnsetDF(signals,t="u"):
    def getOnsets(x):
        spec = madmom.audio.spectrogram.Spectrogram(x, frame_size=300, hop_size=300)
        X=madmom.features.onsets.high_frequency_content(spec)
        return X[0:100] 
    onsets=[]
    for s in signals:
        onset=getOnsets(s)
        onsets.append(onset[0:300])
    
    df=pd.DataFrame(onsets)
    feat_cols=['onset'+str(i) for i in range(df.shape[1])]
    df.columns=feat_cols
    df["label"]=t
    return df

def fitPolyWave(signals,t="unknown_drum",frameLen=2000,hopLen=1600,polyDeg=2,num_feats=100):
    def getFeat(x):
        fs=madmom.audio.signal.FramedSignal(x, sample_rate=40000,
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



