#!/usr/bin/env python
# coding: utf-8

# In[42]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.initializers import glorot_uniform  # Or your initializer of choice
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.callbacks import TensorBoard

import numpy as np
import random
import mir_utils as miru
import sounddevice as sd

sr=40000 #sample rate
input_dim=5000
group_size=100
# this is the size of our encoded representations
encoding_dim = 100  #floats -> compression factor of input_dim/encoding_dim

# this is our input placeholder
input_sample= Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_sample)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='linear')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_sample, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_sample, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy'])

# checkpoint
filepath="./models/linear/weights-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# In[43]:


#takes an audio dict, desired length of samples and split percentage of test & train subsets
#returns x_train,x_test,y_train,y_test
def audioDictToNp(group_size=10,dur=10000,testFraction=0):
    a=miru.loadAudioArrays()
    X=[]
    y=[]
    for key,l in a.items():
            for i in l:
                if len(i)>dur and key!="asdfkick":
                    y.append(key)
                    X.append(i[0:dur])
    X=np.asarray(X)
    y=np.asarray(y)
    if testFraction==0:
        return X,X,y,y
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testFraction, random_state=42)
        return X_train, X_test, y_train, y_test
            
x_train, x_test, y_train, y_test=audioDictToNp(group_size=group_size,dur=input_dim,testFraction=0.1)

print("train,test shapes:",x_train.shape,x_test.shape)


# In[ ]:


def train():
    initial_weights = autoencoder.get_weights()
    k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
    new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
    
    weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
#     autoencoder.set_weights(weights)
    
    autoencoder.fit(x_train, x_train,
                    epochs=1000,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1,
                    #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                    callbacks=callbacks_list,
                   )
train()

# encode and decode some digits
# note that we take them from the *test* set
encoded_samples = encoder.predict(x_test)
decoded_samples = decoder.predict(encoded_samples)
print(autoencoder.summary())


# In[34]:


from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

n = 9 # how many samples we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(50, int(input_dim/50)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_samples[i].reshape(50, int(input_dim/50)))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# In[35]:


from IPython.display import Audio, display
from ipywidgets import widgets

original_widgets = []
for (audio) in x_train[0:10]:
    out = widgets.Output()
    with out:
        display(Audio(data=audio, rate=sr))
    original_widgets.append(out)
oBox=widgets.HBox(original_widgets)


decoded_widgets = []
for (audio) in decoded_samples[0:10]:
    out = widgets.Output()
    with out:
        display(Audio(data=audio, rate=sr))
    decoded_widgets.append(out)
dBox=widgets.HBox(decoded_widgets)

widgets.VBox([oBox,dBox])


# In[ ]:




