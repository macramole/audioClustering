#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:56:13 2017

@author: leandro
"""

#%% Imports

import os
import numpy as np
from math import ceil
import time

import librosa
import librosa.display
import librosa.core

import matplotlib
import matplotlib.pyplot as plt

import vlc

#%% Load Audio

SAMPLE_RATE = 22050
SEGUNDOS_FILA = 1
SIZE_AUDIO_RAW = ceil(SAMPLE_RATE * SEGUNDOS_FILA)


#SELECCION_DIR = "data/sound/pack/drumkits.mp3/"
SELECCION_DIR = "data/sound/else/"
FILE_TYPE=".mp3"

def findMusic(directory):
    musicFiles = []
    
    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/")
        elif file.endswith(FILE_TYPE):
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                print("Skipped:", directory + file)
    
    return musicFiles

#%% Find files

audioFiles = findMusic( SELECCION_DIR )

print("")
print("Encontré",len(audioFiles),"archivos de audio")    

#%% Procesos

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



#%% Testing

#testAudioData, _ = librosa.core.load(audioFiles[0], sr = SAMPLE_RATE)
audioFileToLoad = audioFiles[round( np.random.uniform(0, len(audioFiles)) )]
testAudioData, _ = librosa.core.load( audioFileToLoad, sr = SAMPLE_RATE)

vlc.MediaPlayer(audioFileToLoad).play()


#pStream.write( testAudioData.astype(np.float32).tostring() )


testAudioData = testAudioData[0:SIZE_AUDIO_RAW]
#RMSE
testRMSE = librosa.feature.rmse( testAudioData )
plt.semilogy(testRMSE.T, label='RMS Energy')
plt.xticks([])
plt.xlim([0, testRMSE.shape[-1]])
plt.legend(loc='best')
    
#STFT
testSTFT = librosa.stft( testAudioData )
testSTFTdb = librosa.amplitude_to_db(testSTFT, ref=np.max)
#plt.subplot(4, 2, 1)
librosa.display.specshow(testSTFTdb, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

#MFCC
testMFCC = librosa.feature.mfcc( testAudioData, sr = SAMPLE_RATE )
librosa.display.specshow(testMFCC, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()

#Mel spectogram
testMelSpectogram = librosa.feature.melspectrogram(testAudioData, SAMPLE_RATE)
librosa.display.specshow( librosa.core.power_to_db(testMelSpectogram , ref=np.max), y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()


#Zero crossing
testZeroCrossing = librosa.feature.zero_crossing_rate(testAudioData)


#%% LSTM

from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model

inputs = Input(shape=(timesteps, input_dim))
encoded = LSTM(latent_dim)(inputs)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(input_dim, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)