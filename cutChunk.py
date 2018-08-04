#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 01:08:44 2018

@author: macramole
"""

#%%

from pydub import AudioSegment
import os

#%%

RESULT_PATH = "/home/macramole/Data/trigal"
SPEED = "0.5"
CHUNK_LENGTHS = [100,200,500,800,1000,1200,2000]

#%%

song = AudioSegment.from_mp3("/media/macramole/stuff/Desktop/Sandro - Trigal - x%s.mp3" % SPEED)
song = song.set_channels(1)

#%%

for chunk_length in CHUNK_LENGTHS:
    directory = "%s/x%s_%dms" % (RESULT_PATH, SPEED, chunk_length)
    os.mkdir(directory)
    for i in range(0, len(song), chunk_length):
        song[i:(i+chunk_length)].export("%s/x%s_%dms_%d.wav" % (directory, SPEED, chunk_length, i), format = "wav", parameters = "-ac1")