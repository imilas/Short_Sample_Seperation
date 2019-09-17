from collections import defaultdict
import pickle,dill
import madmom
import random
import numpy as np
import seaborn as sns
import pandas as pd
import scipy, matplotlib.pyplot as plt, sklearn, urllib, IPython.display as ipd
import sounddevice as sd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
import os
import librosa, librosa.display
rootdir = os.getcwd()
#load a sample, if given path, load it,
#if no path but given type, randomly pick one of the type
#else randomly pick type and load one of the type
samples="dk_data"
def loadSample(path="",soundType="",sr=40000):
        if path:
                file=path
                y, sr = librosa.load(path,sr)
        elif soundType:
                path="./%s/%s/"%(samples,soundType,)
                file=random.choice(os.listdir(path))
                y, sr = librosa.load(path+file,sr)
        else:
                soundType=random.choice(os.listdir("./"+samples))
                path="./%s/%s/"%(samplessoundType,)
                file=random.choice(os.listdir(path))
                y, sr = librosa.load(path+file,sr)               
        return y,sr,file,path


#load all samples into a dictionary of arrays
#can load by opening the pickled dictionary or fresh load if added new sounds

def loadAudioArrays(load=True,save=True,path="dk_samples"):
        if load==True:
                try:
                        file=open("audio_dict.dill","rb")
                        f=dill.load(file)
                        return f
                except:
                        print("nothing to load")  
        else:
                # f is a dictionary of lists for all audio files under a folder
                f = defaultdict(list)
                print(rootdir)
                for subdir, dirs, files in os.walk("./dk_data"):
                        print("loading\n\n\n" + subdir) 
                        for file in files: 
                                filepath = subdir + os.sep + file
                                try:
                                        y, sr = librosa.load(filepath,sr=40000)
                                        y=madmom.audio.signal.rescale(y)
                                        y=madmom.audio.signal.trim(y)
                                        yt, index = librosa.effects.trim(y,top_db =40,frame_length=5000, hop_length=50)
                                        yt=librosa.util.normalize(yt)
                                        if(subdir=="/home/amir/mir/t-sne/samples/rims"):
                                        # librosa.output.write_wav(filepath, yt, sr)
                                        # print(librosa.get_duration(y), librosa.get_duration(yt))
                                                sd.play(yt,sr,blocking=True,blocksize=500)
                                        f[subdir.split("/")[-1]].append(y)
                                except:
                                        continue
                if(save):        
                        file=open("audio_dict.dill","wb")
                        dill.dump(f,file)
                return f


#load a n-sized subset of samples longer than dur
def loadAudioSubset(n,dur=1000):
        # f is a dictionary of lists for n audio files under a folder
        f = defaultdict(list)
        for subdir, dirs, files in os.walk(rootdir+"/dk_data"):
                print("loading: "+subdir)
                i=0 
                for file in files: 
                        if i<n:
                                filepath = subdir + os.sep + file
                                y, sr = librosa.load(filepath,sr=40000)
                                y=madmom.audio.signal.rescale(y)
                                y=madmom.audio.signal.trim(y)
                                yt, index = librosa.effects.trim(y,top_db =40,frame_length=5000,
                                                                hop_length=50)
                                yt=librosa.util.normalize(yt)
                                if len(yt)>dur:
                                        f[subdir.split("/")[-1]].append(y)
                                        i+=1
                        else:
                                break
        return f

#given a dictionary of sounds play all samples
def playDict(f):
        for key,l in f.items():
                print(key)                                                                                                                       
                for i in l:
                        trimmed,index=librosa.effects.trim(i,top_db =45,
                        frame_length=2, hop_length=1) 
                        print(i.shape,trimmed.shape,index)                                                                                                                      
                        sd.play(trimmed,40000,blocking=True,blocksize=1000)

#makes a t-sne for any dataframe of features
def plotTSNE(df,perp=2):
    data_subset=df.loc[:,df.columns!="label"].values
    data_subset=np.absolute(data_subset)
    tsne = TSNE(n_components=2, verbose=0, perplexity=perp, n_iter=3000)
    tsne_results = tsne.fit_transform(data_subset)
    time_start = time.time()
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(8,5))

    flatui = ["#000000", "#3498db", "#00AA00", "#e74c3c", "#FF49FF", "#2ecc71"]
    sns.set_palette(sns.color_palette(flatui))

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="label",
        data=df,
        legend="full",
        alpha=1
    )
    plt.show()

def playSample(sample,sr=40000):
    sd.play(sample,sr,blocking=True)

if __name__=="__main__":
        # f=loadAudioArrays()
        f=loadAudioSubset(10)
        playDict(f)   
