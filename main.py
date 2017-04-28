# -*- coding: utf-8 -*-

#%% Imports

import os
import numpy as np
from math import ceil
import time

import librosa

import matplotlib
import matplotlib.pyplot as plt

#%% Load Audio

SAMPLE_RATE = 22050
SIZE_STFT = 45100 #esto seguramente se pueda calcular pero bue

SEGUNDOS_FILA = 1
SELECCION_DIR = "data/sound/pack/drumkits/"

def doSTFT(data):
    D = librosa.stft(data)
    D = np.abs(D)
#    return D.reshape(1, D.shape[0] * D.shape[1] )
    return D.reshape(D.shape[0] * D.shape[1] )

def findMusic(directory):
    musicFiles = []
    
    for file in os.listdir(directory):
        if os.path.isdir(directory + file):
            musicFiles += findMusic(directory + file + "/")
        elif file.endswith(".wav"):
            musicFiles.append( directory + file )
        else:
            if not file.endswith(".asd"):
                print("Skipped:", directory + file)
    
    return musicFiles

#%% Find files

audioFiles = findMusic( SELECCION_DIR )

print("")
print("Encontré",len(audioFiles),"archivos de audio")    


#%% FFT Process

import sys

#matrixAudioDataNoFFT = np.empty((0,ceil(SAMPLE_RATE * SEGUNDOS_FILA)), np.int16)
#matrixAudioData = np.empty((0, SIZE_STFT), np.float32)
matrixAudioData = np.empty((0, SIZE_STFT + (SAMPLE_RATE * SEGUNDOS_FILA) ), np.float32)

count = 0
COUNT_NOTICE = 200

tic = time.clock()

for file in audioFiles:
    try:
        print(".", end="")

        tmpAudioData, tmpSampleRate = librosa.core.load(file, sr = SAMPLE_RATE)
        
        tmpAudioData.resize(ceil(SAMPLE_RATE * SEGUNDOS_FILA))

        stft = doSTFT(tmpAudioData)
    
        stftYRawData = np.concatenate( (stft, tmpAudioData ) )
        stftYRawData = stftYRawData.reshape(1, stftYRawData.shape[0])

        matrixAudioData = np.concatenate((matrixAudioData, stftYRawData ), axis = 0 )

        count += 1

        if count % COUNT_NOTICE == 0:
            print("")
            print("[", count, "/", len(audioFiles), "]")
    except:
        print(file, "[FAIL]")
    
#        if count >= 100:
#            break
        
print("")
print("Matriz final:", matrixAudioData.shape)

toc = time.clock()
print("time:", toc - tic)

#%% Guardar para no tener que procesar mil veces

np.save("matrixAudioDataWithRawData", matrixAudioData)

#matrixAudioData = np.load("matrixAudioData.npy")
#matrixAudioData.shape

#%% Principal component analysis

from sklearn.decomposition import PCA

tic = time.clock()

pca = PCA(n_components=800) #500
pca.fit(matrixAudioData)
print("Variance explained:", pca.explained_variance_ratio_.sum())
matrixAudioDataTransformed = pca.transform(matrixAudioData)

toc = time.clock()

# print(matrixAudioDataTransformed[0])
print("shape transformed:", matrixAudioDataTransformed.shape)

print("time:", toc - tic)

#%% Cluster Jerárquico

from scipy.cluster import hierarchy as h
from scipy.spatial import distance as dist

distanceFunction = 'cosine' #canberra, cityblock, braycurtis, euclidean
linkageType = 'average' #single, complete, weighted, average

tic = time.clock()

distanceMatrix = dist.pdist(matrixAudioDataTransformed, distanceFunction) 
clusters = h.linkage(distanceMatrix, linkageType)
c,d=h.cophenet(clusters, distanceMatrix) #factor cofonético

toc = time.clock()

print("distancia:",distanceFunction)
print("linkage:",linkageType)
print("cofonetica:",c)
print("time:", toc - tic)

#%% Dendograma

THRESHOLD = 0.942


cutTree = h.cut_tree(clusters, height= THRESHOLD)
# minutos = 0
# segundos = 0
# for label in cutTree:
#     print(str(minutos) + ":" + str(segundos), str(label))
#     segundos += SEGUNDOS_FILA
#     if ( segundos >= 60 ):
#         minutos += 1
#         segundos -= 60

# print(clusters.shape)
# print(clusters)
# exit()

#matrixAudioDataAndCluster = np.hstack((matrixAudioDataTransformed, cutTree))

cantClusters = np.unique(cutTree).size
print("Cantidad de clusters:", cantClusters)

plt.figure(figsize=(15, 3))
dn = h.dendrogram(clusters, color_threshold = THRESHOLD, 
                  show_leaf_counts = True, no_labels = True, 
                  p = 200, truncate_mode = "lastp")

# draw vertical line from (70,100) to (70, 250)
plt.axhline(y=THRESHOLD, c='k', lw = 1, linestyle = "dotted")
plt.show()

#%% Histograma

plt.figure(figsize=(12, 5))
n = plt.hist(cutTree, bins=cantClusters)[0]
plt.show()
print("Count:")
print(n)

#%% Multi-dimensional scaling

#check http://scikit-learn.org/stable/auto_examples/manifold/plot_mds.html

from sklearn.manifold import MDS 
from sklearn.metrics import pairwise_distances

tic = time.clock()

#similarities = pairwise_distances(matrixAudioDataTransformed, metric = distanceFunction, n_jobs = -1)
similarities = pairwise_distances( dist.squareform(distanceMatrix), n_jobs = -1)

mds = MDS(n_components=2, dissimilarity="precomputed", n_jobs=-1) #max_iter=3000, eps=1e-9, 
positions = mds.fit(similarities).embedding_

toc = time.clock()

#positions = mds.fit(distanceMatrix).embedding_
print("Positions shape:", positions.shape)
print("time:", toc - tic)

#%% Test de independencia

COUNT_TOP = 10
fileNameCount = {}
fileNameEquivalentes = {
    "bd" : "bassdrum",
    "toms" : "tom",
    #"tom h" : "tom",
    #"tom l" : "tom",
    #"tom m" : "tom",
    "crash" : "cymbal",
    "rim shot" : "rimshot",
    "ride" : "cymbal",
    "hat closed" : "cymbal",
    "hat open" : "cymbal",
    "conga h": "conga",
    "conga l": "conga",
}
audioFilesClass = []

for audioFile in audioFiles:
    fileName = audioFile[ audioFile.rfind("/") + 1 : audioFile.rfind(".wav") ]
    fileName = fileName.replace(".wav", "")
    
    if ( fileName.find("-") != -1 ):
       fileName = fileName[0:fileName.rfind("-")]
    
    if ( fileName.find("_") != -1 ):
       fileName = fileName[0:fileName.rfind("_")]
    
    fileName = fileName.lower()
    
    if ( fileName in fileNameEquivalentes ):
        fileName = fileNameEquivalentes[fileName]
    
    if ( not fileName in fileNameCount ):
        fileNameCount[fileName] = 0
    
    audioFilesClass.append(fileName)
    fileNameCount[fileName] = fileNameCount[fileName] + 1

fileNameCountTop = {}
    
sumCount = 0
    
for i in range(0,COUNT_TOP):
    maxCount = -1
    maxIndex = None
    
    for fileName in fileNameCount:
        if ( fileNameCount[fileName] > maxCount ):
            maxCount = fileNameCount[fileName]
            maxIndex = fileName
    
    fileNameCountTop[maxIndex] = maxCount
    fileNameCount[maxIndex] = -1
    sumCount = sumCount + maxCount
    
print(sumCount)
print(sumCount / len(audioFiles))
fileNameCountTop

#%% Matriz de contingencia

matrizContingencia = np.zeros(shape=( COUNT_TOP, cantClusters - 1 ))
#matrizContingencia = np.zeros(shape=( COUNT_TOP, 2 ))

for i in range(0, len(audioFilesClass)):
    if ( cutTree[i] == cantClusters - 1 ):
        continue
    clase = audioFilesClass[i]
    if ( clase in fileNameCountTop ):
        cluster = cutTree[i]
        #cluster = 0
        #if ( cutTree[i] != 1 ):
        #    cluster = 1
        indexClase = list(fileNameCountTop).index(clase)
        matrizContingencia[ indexClase, cluster ] = matrizContingencia[ indexClase, cluster ] + 1

print(fileNameCountTop)
matrizContingencia

from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(matrizContingencia)

#%% Espectrogramas Promedio por cluster

import scipy.stats

espectrogramasPromedio = []
espectrogramasMediana = []

for cluster in range(0, cantClusters):
    npAudioFiles = np.array( audioFiles )[ output[:,2] == str(cluster) ]
    espectrogramasPromedioPre = []
    # espectrogramasPromedio[0] = np

    for i in range(0, npAudioFiles.size ):
    #     
        y, sr = librosa.load(npAudioFiles[i], sr = SAMPLE_RATE)

        y.resize(ceil(SAMPLE_RATE * SEGUNDOS_FILA))

        D = librosa.stft(y)

        spectrogramData = librosa.logamplitude(np.abs(D)**2, ref_power=np.max)

        espectrogramasPromedioPre.append(spectrogramData)


    #     plt.title('Power spectrogram')
    #     plt.colorbar(format='%+2.0f dB')
    #     matplotlib.pyplot.savefig(ESPECTROGRAMA_SAVE_TO + str(i) + ".png", bbox_inches='tight')
    #     plt.close(fig)
    #     plt.show()
    #     break

    espectrogramasPromedio.append( np.average( espectrogramasPromedioPre, axis = 0 ) )
    espectrogramasMediana.append( np.median( espectrogramasPromedioPre, axis = 0 ) )
    
for cluster in range(0, cantClusters):
    fig = plt.figure(figsize=(6, 5))
    librosa.display.specshow(espectrogramasPromedio[cluster],y_axis='log', x_axis='time')
    plt.title('Cluster ' + str(cluster))

plt.show()



espectrogramasPromedio = []

for i in range(0,cantClusters) :
    
    average = np.average( librosa.logamplitude( matrixAudioData [ output[:,2] == "0", :], ref_power=np.max ), axis = 1 )
    espectrogramasPromedio.append( average )

espectrogramasPromedio



import librosa.display

# ESPECTROGRAMA_SAVE_TO = "results/espectrogramas/"

# for i in range(0, len(audioFiles)): #guarda el range
fig = plt.figure(figsize=(16, 5))
librosa.display.specshow( espectrogramasPromedio[0] ,y_axis='log', x_axis='time')
plt.title('Power spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.show()
#matplotlib.pyplot.savefig(ESPECTROGRAMA_SAVE_TO + str(i) + ".png", bbox_inches='tight')
#plt.close(fig)

#%% Output

audioFilesForExport = list( map( lambda x : x[25:], audioFiles ) )
output = np.c_[ positions, cutTree, audioFilesForExport ]


## Para visualizar 

np.savetxt("audioClusteringResult.tsv", 
           output, 
           fmt = "%s", 
           header = "x\ty\tcluster\tfile",
           delimiter = "\t") 

## Matriz de contingencia

np.savetxt("matrizContingencia-observado.tsv", 
           matrizContingencia, 
           fmt = "%s", 
           delimiter = "\t") 
np.savetxt("matrizContingencia-esperado.tsv", 
           expected, 
           fmt = "%s", 
           delimiter = "\t") 

## Espectrogramas

import librosa.display

ESPECTROGRAMA_SAVE_TO = "results/espectrogramas/"

for i in range(0, len(audioFiles)): #guarda el range
    fig = plt.figure(figsize=(6, 5))
    y, sr = librosa.load(audioFiles[i], sr = SAMPLE_RATE)
    
    y.resize(ceil(SAMPLE_RATE * SEGUNDOS_FILA))
    
    D = librosa.stft(y)
    librosa.display.specshow(librosa.logamplitude(np.abs(D)**2,
                                               ref_power=np.max),
                          y_axis='log', x_axis='time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    matplotlib.pyplot.savefig(ESPECTROGRAMA_SAVE_TO + str(i) + ".png", bbox_inches='tight')
    plt.close(fig)
