#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:11:39 2018

@author: leandro
"""
#%%

import librosa
import matplotlib.pyplot as plt
from librosa.display import specshow
import numpy as np

#%%

sonidos = {
    "Kick" : [ 
        "/home/leandro/Data/drumkits/Roland TR-808/Bassdrum-05.wav",
        "/home/leandro/Data/drumkits/Roland TR-909/Bassdrum-02.wav",
        "/home/leandro/Data/drumkits/Boss DR-110/Bassdrum.wav",    
    ],
    
    "Snare" : [
        "/home/leandro/Data/drumkits/Roland TR-808/Snaredrum.wav",
        "/home/leandro/Data/drumkits/Roland TR-909/Snaredrum.wav",
        "/home/leandro/Data/drumkits/Boss DR-110/Snaredrum.wav"
    ],
    
    "Crash" : [
        "/home/leandro/Data/drumkits/Roland TR-808/Crash-01.wav",
        "/home/leandro/Data/drumkits/Roland TR-909/Crash.wav",
        "/home/leandro/Data/drumkits/Boss DR-110/Crash.wav"
    ]
}

#%%

def tosize(a, n):
   a = np.asarray(a)
   x = np.zeros(n, dtype=a.dtype)
   m = min(n, len(a))
   x[:m] = a[:m]
   return x

#%%

index = 1

for c in sonidos:
    cat = sonidos[c]
    for i in range(0, len(cat)):
        sound, sr = librosa.load(cat[i])
        sound = tosize(sound, 8200)
        sound= librosa.amplitude_to_db( librosa.stft(sound), ref=np.max )
        
        librosa.display.specshow(sound, y_axis='log', x_axis="time")
    #    plt.colorbar(format='%+2.0f dB')
        title = "%s - %s" % (c,cat[i].split("/")[5])
        plt.title(title)
        index += 1
        plt.savefig( "Spectrograms/%s.jpg" % title )
