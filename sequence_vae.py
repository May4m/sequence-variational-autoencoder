'''
 Implementation of a seq-2-seq Variational Autoendocer
 Inspired by Neural Machine Translation

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

from utils import dataset, sin_signal_freq, draw_latent_frequency

from datetime import datetime

import matplotlib.pyplot as plt


import numpy as np
from scipy.stats import norm
from scipy.signal import square

from sklearn.model_selection import train_test_split
from keras.layers import (Input, Dense, Lambda, Layer, LSTM, Dropout,
			  TimeDistributed, Embedding, RepeatVector)
from keras.callbacks import TensorBoard, Callback
from keras.optimizers import RMSprop, Adagrad
from keras.models import load_model, Model
from keras import backend as K
from keras import metrics

import numpy as np
import pickle



# Model Parameter

seq_length = 309
feature_size = 2
batch_size = 100
original_dim = 784
latent_dim = 2
epochs = 100
epsilon_std = 1.0


model_shape={'lstm': 100, 'td-dense': 50, 'dense': 512, 'out': 50}
#feature_size = 2
input_shape = (120, 2, 10)  # (batch, features, seq length)


state = None
main_input = None


def log_dir():
    """
    Inner function of create_model()
    Creates unique datetime string for training dir location
    """
    date = datetime.now()
    r = '{0}_{1:02d}_{2:02d}_'.format(date.year, date.month, date.day)
    r += '{0:02d}_{1:02d}_{2:02d}'.format(date.hour, date.minute, date.second)
    return r


def roll(x):
    shifted = np.zeros_like(x)
    shifted[:,:-1,:] = x[0:,1:,0:]
    return shifted


def SequenceEncoder():
    """
    LSTM encoder with the last layer prediction as the input to the dense layer which provides
    variational parameters `guassian mean` and `guassian variance`
    """

    global state, main_input


    main_input = Input(shape=(None, feature_size))
    seq_1 = LSTM(model_shape['lstm'])(main_input)
    
    
    hidden_1 = Dense(model_shape['dense'], activation='relu')(seq_1)
    dropout = Dropout(0.0)(hidden_1)
    
    
    z_mean, z_log_var = (Dense(latent_dim, kernel_initializer='random_uniform')(dropout),
                         Dense(latent_dim, kernel_initializer='random_uniform')(dropout))
    return z_mean, z_log_var, main_input


def SequenceDecoder(z, generator=False):
    """
    LSTM decoder conditioned on the output of the dense layer symetrical to the encoder
    """

    hidden_1 = Dense(model_shape['dense'], activation='relu')(z)
    dropout = Dropout(0.0)(hidden_1)
    
    state = Dense(feature_size, activation='relu')(dropout)
    lstm = LSTM(feature_size, return_sequences=True, activation='linear')(main_input, [state, state])
    return lstm


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


def final_output(args):
    z_mean, z_log_var = args
    return z_mean + K.exp(z_log_var)



# Custom loss layer
class VariationalLayer(Layer):
    """
    "Bottle Neck Layer". This layer provides the kl-divergence loss function
    """

    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.iter = 0
        self.target = Input(shape=(None, feature_size))  # original time sequence shift once
        super(VariationalLayer, self).__init__(**kwargs)
        

    def vae_loss(self, x, x_decoded_mean):
        """
        kl-divergence loss plus reconstrucion loss 
        """

        self.xent_loss = K.sum(metrics.mean_squared_error(self.target, x_decoded_mean), axis=1)  # sum error over time
        self.kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(self.xent_loss + self.kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

    

class Anealing(Callback):
    """
    scaling the kl-divergence loss by anealing it with a smooth function: tanh in this instance.
    the tanh in this case depends on the number of epochs for the the rate of growth.
    """

    def tanh_aneal(self, iter):
        return (K.tanh((iter - 5) / 0.25) + 1) / 2

    def on_batch_end(self, epoch, logs={}):
        iter_num = K.tf.cast(self.model.optimizer.iterations, K.tf.float32)
        aneal_factor = K.eval(self.tanh_aneal(iter_num))
        self.model.get_layer("loss_layer").xent_loss *= aneal_factor

        
z_mean, z_log_var, x = SequenceEncoder()


# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
x_decoded_mean = SequenceDecoder(z)

vae_layer = VariationalLayer(name='loss_layer')
y = vae_layer([x, x_decoded_mean])
vae = Model([x, vae_layer.target], y)
vae.compile(optimizer=Adagrad(lr=0.001), loss=None)


std_output = Lambda(final_output, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x,  std_output)


print(vae.summary())
print(dataset.shape)
print("[ INFO ] Ready to train model")



shifted_dataset = roll(dataset)
vae.fit([dataset, shifted_dataset],
        shuffle=True,
        epochs=1,
        validation_split=0.1,
        batch_size=128,
        callbacks=[TensorBoard(log_dir='/tmp/dumps/seq_model_' + log_dir()),
                   Anealing()])



encoder.save('encoder.h5')
draw_latent_frequency(encoder)
