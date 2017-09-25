#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:04 2017

@author: leandro
"""

import utils
import numpy as np
import matplotlib.pyplot as plt

#SELECCION_DIR = "data/sound/pack/drumkits.mp3/"
SELECCION_DIR = "data/sound/else/"
FILE_TYPE=".mp3"

#%% Prep data

audioFiles = utils.findMusic(SELECCION_DIR, FILE_TYPE)
print( "Found", len(audioFiles), "files" )

matrixAudioData = utils.getAudioData(audioFiles)

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
from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers


activationFunction = "linear"

inputData = Input(shape=(matrixAudioData.shape[1],))
encoded = Dense(128, activation=activationFunction )(inputData)
encoded = Dense(64, activation=activationFunction)(encoded)
encoded = Dense(32, activation=activationFunction)(encoded)
encoded = Dense(16, activation=activationFunction)(encoded)
encoded = Dense(8, activation=activationFunction)(encoded)
encoded = Dense(2, activation=activationFunction)(encoded)
#encoded = Dense(2, activation=activationFunction)(inputData)

decoded = Dense(8, activation=activationFunction)(encoded)
decoded = Dense(16, activation=activationFunction)(decoded)
decoded = Dense(32, activation=activationFunction)(decoded)
decoded = Dense(64, activation=activationFunction)(decoded)
decoded = Dense(128, activation=activationFunction)(decoded)
decoded = Dense(matrixAudioData.shape[1], activation=activationFunction)(decoded)
#decoded = Dense(matrixAudioData.shape[1], activation=activationFunction)(encoded)


autoencoder = Model(inputData, decoded)
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
autoencoder.compile(optimizer= sgd, loss='hinge')


minValue = np.min(matrixAudioData)
maxValue = np.max(matrixAudioData)
scaledMatrixAudiodata = scale(matrixAudioData, minValue, maxValue)

#tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

history = autoencoder.fit(scaledMatrixAudiodata, scaledMatrixAudiodata,
                epochs=2000,
                batch_size=256,
                shuffle=True)
#                callbacks = [tbCallBack] )
#                validation_data=(x_test, x_test))

plt.plot(history.history["loss"])

    #%% Save model
autoencoder.save("feedforward_autoencoder.model")
autoencoderJSON = autoencoder.to_json()
with open("feedforward_autoencoder.json", "w") as json_file:
    json_file.write(autoencoderJSON )


#%% "test" model
    
#test = autoencoder.predict( scale(matrixAudioData[0,:], minValue, maxValue).reshape((1,880)) )
#test = unScale( test, minValue, maxValue )
#
#matrixAudioData[0,:]
    
#%% Inspecciono
        
activaciones = utils.get_activations( autoencoder, scaledMatrixAudiodata, True )
activaciones = np.array(activaciones[6])
#activaciones = np.power(activaciones,-20)
#activaciones = np.power(activaciones,11)
plt.scatter( activaciones[:,0], activaciones[:,1] )

#%% output
audioFilesForExport = list( map( lambda x : x[len(SELECCION_DIR):], audioFiles ) )
output = np.c_[ activaciones, np.repeat(1,len(audioFilesForExport)), audioFilesForExport ]

np.savetxt("audioDataMFCCElse.tsv", 
           output, 
           fmt = "%s", 
           header = "x\ty\tcluster\tfile",
           delimiter = "\t") 