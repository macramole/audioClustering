#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 16:31:04 2018

@author: leandro
"""

#%% Load dataset

import json
import pandas as pd

with open("/home/leandro/Data/drumkits/session.stft.dbscan.json") as f:
    df = json.load(f)
    strCSV = df["tsv"]
    strCSV = strCSV.split("|")
    
    listCSV = []
    for row in strCSV:
        listCSV.append( row.split(",") )
    
    listCSV.pop()
    
    df = pd.DataFrame(listCSV)

#%% Plot
    
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

from plotly.offline import plot

#py.sign_in('macramole', 'kJrcOfunOIu2uMhPyQFc')

data = [
    go.Scatter( x = df[0], y = df[1], mode = "markers" )
]
#plot(data)
layout = go.Layout(title="asd", width=1024, height=768)
fig = go.Figure(data = data, layout = layout)

plot(fig, image="svg",  image_filename="plot", image_width=1024, image_height=768)

