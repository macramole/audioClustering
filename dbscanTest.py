#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:03:49 2018

@author: leandro
"""

import utils
import numpy as np

audioFiles = utils.findMusic( "/home/leandro/Data/drumkits/", "wav")
audioData = utils.getAudioData( audioFiles, features = "stft" )

audioData = utils.doPCA(audioData)
#np.save("drumkit.stft.pca",audioData)

from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import pandas as pd

utils.doDistanceMatrix(audioData)
tsne = utils.doTSNE( audioData, 2 )


db = DBSCAN( eps=2.8, min_samples=5, metric="euclidean" )
dbFit = db.fit( tsne )

pd.Series(dbFit.labels_).value_counts()
np.unique(dbFit.labels_).shape[0]

plt.hist(dbFit.labels_)


