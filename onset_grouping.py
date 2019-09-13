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


df=pd.DataFrame()
imp.reload(ff)
for key,signals in a.items():
    print("features from %s"%key)
    # chunk=ff.fitPolyWave(signals)
    chunk=ff.fitMels(signals,t=key,num_feats=20)
    df=pd.concat([df,chunk])

df=df.fillna(0)
#find most useless features
#df.astype(bool).sum(axis=0).sort_values()

rndperm = np.random.permutation(df.shape[0])
#t-sne
miru.plotTSNE(df,perp=10)