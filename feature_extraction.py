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


keys=['kick','rim','tom_high','snare','hihat_open']

a=miru.loadAudioArrays()
df=pd.DataFrame()
for key in keys:
    print(key)
    signals=a[key]
    chunk=ff.fitFreq(signals,t=key)
    df=pd.concat([df,chunk])

df=df.fillna(0)
rndperm = np.random.permutation(df.shape[0])
df.to_csv("feat_frequency_bins.csv",index=False)
#t-sne
miru.plotTSNE(df,perp=20)
plt.show()