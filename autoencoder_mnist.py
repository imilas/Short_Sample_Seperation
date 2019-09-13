#!/usr/bin/env python
# coding: utf-8

# In[49]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.initializers import glorot_uniform  # Or your initializer of choice
from keras.datasets import mnist
import numpy as np
import random
from keras.callbacks import TensorBoard

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()


# In[50]:


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


# In[51]:


print (x_train.shape)
print (x_test.shape)

Wsave = autoencoder.get_weights()
for a in Wsave:
    print(a.shape)
for layer in autoencoder.layers:
     print(layer.get_output_at(0).get_shape().as_list())

print(autoencoder.summary())


# In[52]:


x_train=x_train[0:50]
x_test=x_train[0:10]
print (x_train.shape)
print (x_test.shape)
def train():
    initial_weights = autoencoder.get_weights()
    weights = [glorot_uniform(seed=random.randint(0, 1000))(w.shape) if w.ndim > 1 else w for w in autoencoder.get_weights()]
    autoencoder.set_weights(initial_weights)

    autoencoder.fit(x_train, x_train,
                    epochs=1000,
                    batch_size=10,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    verbose=1,
                    callbacks=[TensorBoard(log_dir='/tmp/autoencoder')]
                   )
train()


# In[53]:



# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

from keras.datasets import mnist
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

