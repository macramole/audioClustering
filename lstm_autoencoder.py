#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:04 2017

@author: leandro
"""

import utils
import numpy as np

#import PyQt5
#import matplotlib

#%matplotlib qt5

import matplotlib.pyplot as plt

SELECCION_DIR = "data/sound/drumkit/"
SELECCION_DIR = "data/sound/pack/drumkits.mp3/"
SELECCION_DIR = "data/sound/else/"
FILE_TYPE=".mp3"

#%% Prep data

audioFiles = utils.findMusic(SELECCION_DIR, FILE_TYPE)
print( "Found", len(audioFiles), "files" )

#%% Process data

matrixAudioData = utils.getAudioData(audioFiles, superVector = False, qtyFilesToProcess = 100)
#shape : (100, 20, 44) 100 canciones, 44 tiempo, 20 mfcc
#%% Save/Load prep data

#utils.saveAudioData(matrixAudioData, "audioDataMFCCelse")
matrixAudioData = utils.loadAudioData("audioDataMFCCelse.npy")


#%% Scale entre 0 y 1
# Esto da unos errores medio raros de cálculo

def scale(data, min = None, max = None):
    if min == None:
        min = np.min(data)
        
    if max == None:
        max = np.max(data)
    
    return np.divide( ( data - min ) , ( max - min) )

def unScale(scaledData, min, max):
    return ( scaledData  * ( max - min) ) + min

#%% Autoencoder
import keras
from keras.layers import Input, LSTM, Dense, RepeatVector
from keras.models import Model
from keras import regularizers
from keras import optimizers

activationFunction = "sigmoid"
latent_dim = 2
timesteps = matrixAudioData.shape[2]
input_dim = matrixAudioData.shape[1]

inputs = Input( shape=(timesteps, input_dim) )
encoded = LSTM( latent_dim, activation=activationFunction )(inputs)

decoded = RepeatVector( timesteps )( encoded )
decoded = LSTM( input_dim, return_sequences = True )(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model( inputs, encoded )

#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
sequence_autoencoder.compile(optimizer= sgd, loss='hinge')


minValue = np.min(matrixAudioData)
maxValue = np.max(matrixAudioData)
scaledMatrixAudiodata = scale(matrixAudioData, minValue, maxValue)

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

history = sequence_autoencoder.fit(scaledMatrixAudiodata, scaledMatrixAudiodata.reshape(100,44,20),
                epochs=1000,
                batch_size=10,
                shuffle=True)
#                callbacks = [tbCallBack] )
#                validation_data=(x_test, x_test))

plt.plot(history.history["loss"])

    #%% Save model
sequence_autoencoder.save("LSTM_autoencoder.model")
sequence_autoencoderJSON = sequence_autoencoder.to_json()
with open("LSTM_autoencoder.json", "w") as json_file:
    json_file.write(sequence_autoencoderJSON )


#%% "test" model
    
#test = autoencoder.predict( scale(matrixAudioData[0,:], minValue, maxValue).reshape((1,880)) )
#test = unScale( test, minValue, maxValue )
#
#matrixAudioData[0,:]
    
#%% Inspecciono
        
activaciones = utils.get_activations( sequence_autoencoder, scaledMatrixAudiodata.reshape(100,44,20), True )
activaciones = np.array(activaciones[1])
#activaciones = np.power(activaciones,-20)
#activaciones = np.power(activaciones,0.01)

plt.scatter( activaciones[:,0], activaciones[:,1] )

#%% output
audioFilesForExport = list( map( lambda x : x[len(SELECCION_DIR):], audioFiles[0:100] ) )
output = np.c_[ activaciones, np.repeat(1,len(audioFilesForExport)), audioFilesForExport ]

np.savetxt("tsvs/LSTM_autoencoder-mfcc-drums.tsv", 
           output, 
           fmt = "%s", 
           header = "x\ty\tcluster\tfile",
           delimiter = "\t") 