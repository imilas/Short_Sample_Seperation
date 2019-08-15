import os
from collections import defaultdict
import librosa
import pickle,dill
import madmom
import sounddevice as sd
import random

rootdir = os.getcwd()
#load a sample, if given path, load it,
#if no path but given type, randomly pick one of the type
#else randomly pick type and load one of the type
def loadSample(path="",soundType="",sr=40000):
        if path:
                file=path
                y, sr = librosa.load(path,sr)
        elif soundType:
                path="./samples/%s/"%(soundType,)
                file=random.choice(os.listdir(path))
                y, sr = librosa.load(path+file,sr)
        else:
                soundType=random.choice(os.listdir("./samples"))
                path="./samples/%s/"%(soundType,)
                file=random.choice(os.listdir(path))
                y, sr = librosa.load(path+file,sr)               
        return y,sr,file,path

#load all samples into a dictionary of arrays
#can load by opening the pickled dictionary or fresh load if added new sounds
def loadAudioArrays(save=True):
        if(save):
                file=open("audio_dict.dill","rb")
                f=dill.load(file)
                return f
        else:  
        # f is a dictionary of lists for all audio files under a folder
                f = defaultdict(list)
                for subdir, dirs, files in os.walk(rootdir+"/samples"):
                        print("loading\n\n\n" + subdir) 
                        for file in files: 
                                filepath = subdir + os.sep + file
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
        if(save):        
                file=open("audio_dict.dill","wb")
                dill.dump(f,file)
        return f

#given a dictionary of sounds play all samples
def playDict(f):
        import sounddevice as sd
        import numpy as np
        for key,l in f.items():
                print(key)                                                                                                                       
                for i in l:
                        trimmed,index=librosa.effects.trim(i,top_db =45,
                        frame_length=2, hop_length=1) 
                        print(i.shape,trimmed.shape,index)                                                                                                                      
                        sd.play(trimmed,40000,blocking=True,blocksize=1000)
        # for l in f["kicks"]:
        #         # print(key)                                                                                                                      
        #         sd.play(l,40000,blocking=True,blocksize=1000)

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

    plt.figure(figsize=(5,3))

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

if __name__=="__main__":
        f=loadAudioArrays()
        playDict(f)   
