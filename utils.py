#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 12:47:43 2017

@author: leandro
"""

import librosa
import numpy as np
import os 
from math import ceil
import time

SAMPLE_RATE = 22050
SEGUNDOS_FILA = 1
SIZE_AUDIO_RAW = ceil(SAMPLE_RATE * SEGUNDOS_FILA)

def findMusic(directory, fileType):
    musicFiles = []
    
    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/")
        elif file.endswith(fileType):
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                print("Skipped:", directory + file)
    
    return musicFiles

SIZE_STFT = 45100 #esto seguramente se pueda calcular pero bue
def doSTFT(data):
    D = librosa.stft(data)
    D = np.abs(D)
    return D.reshape(1, D.shape[0] * D.shape[1] )
#    return D.reshape(D.shape[0] * D.shape[1] )

SIZE_ZERO_CROSSING_RATE = 22 #esto seguramente se pueda calcular pero bue
def getZeroCrossingRate(data):
    zc = librosa.feature.zero_crossing_rate(data)
    return zc.reshape( zc.shape[0] * zc.shape[1] )

SIZE_RMSE = 22 #esto seguramente se pueda calcular pero bue
def getRMSE(data):
    rmse = librosa.feature.rmse(data)
    return rmse.reshape( rmse.shape[0] * rmse.shape[1] )

SIZE_MFCC = 440 #esto seguramente se pueda calcular pero bue
def getMFCC(data):
    mfcc = librosa.feature.mfcc(data, sr = SAMPLE_RATE)
    return mfcc.reshape( mfcc.shape[0] * mfcc.shape[1] )


def getAudioData( audioFiles, qtyFilesToProcess = None ):
    count = 0
    countFail = 0
    COUNT_NOTICE = 200
    COUNT_FAIL = 20
    
    listAudioData = []
    
    tic = time.clock()
    
    if qtyFilesToProcess == None:
        qtyFilesToProcess = len(audioFiles)
    
    for i in range(0, qtyFilesToProcess):
        try:
            file = audioFiles[i]
            print(".", end="")
    
            tmpAudioData, tmpSampleRate = librosa.core.load(file, sr = SAMPLE_RATE)
            
            tmpAudioData.resize(SIZE_AUDIO_RAW)
    
            mfcc = getMFCC(tmpAudioData)
#            stft = doSTFT(tmpAudioData)
        
#            listAudioData.append( stft )
            listAudioData.append( mfcc )
    
            count += 1
    
            if count % COUNT_NOTICE == 0:
                print("")
                print("[", count, "/", len(audioFiles), "]")
            
        except:
            countFail += 1
            print(file, "[FAIL]")
            
            if countFail >= COUNT_FAIL:
                break
        
    matrixAudioData = np.array(listAudioData, dtype=np.float32)
#    matrixAudioData = matrixAudioData.squeeze(1)
    
    print("")
    print("Matriz final:", matrixAudioData.shape)
    
    toc = time.clock()
    print("time:", toc - tic)
    return matrixAudioData
    
def saveAudioData( matrixAudioData, filename ):
    np.save(filename, matrixAudioData)
    
def loadAudioData( filename ):
    return np.load(filename)

#matrixAudioData = 
#matrixAudioData.shape
    
#%% functiones para inspeccionar activaciones

import keras.backend as K

def get_activations(model, model_inputs, print_shape_only=False, layer_name=None):
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()