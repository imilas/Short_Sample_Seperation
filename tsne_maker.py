import numpy as np
import pandas as pd
import scipy, matplotlib.pyplot as plt
import scipy
import mir_utils as miru

df=pd.read_csv("feat_frequency_bins.csv")
#t-sne 
miru.plotTSNE(df,perp=60)
plt.show()