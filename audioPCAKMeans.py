from scipy.io import wavfile
import numpy as np
from math import ceil
import matplotlib.pyplot as plt

# audioFile = wavfile.read("audio_44100_16.wav");
# audioFile = wavfile.read("audio_22050_16.wav");
# audioFile = wavfile.read("audio_22050_8.wav");
# audioFile = wavfile.read("audio_22050_8_mono.wav");
# sampleRate, audioData = wavfile.read("data/sound/muchachita_22050_16_mono.wav")
# sampleRate, audioData = wavfile.read("data/sound/audio_22050_16_mono.wav")
sampleRate, audioData = wavfile.read("data/sound/smell.wav")
# sampleRate, audioData = wavfile.read("data/sound/blue_monday.wav")
#sampleRate, audioData = wavfile.read("data/sound/audio_22050_16_mono.wav")
#sampleRate, audioData = wavfile.read("data/sound/baby.wav")

#agrego silencio al final para que divida bien
#BPM = 130 blue_monday
#BPM = 90 audio brasilero
BPM = 116 #smell
# BPM = 98 baby

SEGUNDOS_FILA = ceil( 60/BPM ) * 4 #el multiplicador final serían los compaces
SEGUNDOS_FILA = 0.2

print("Tomando",SEGUNDOS_FILA,"segundos")
print("Llenando de silencio al final para que me quede cortable...")

newDataSize = audioData.size
while newDataSize % sampleRate != 0:
    newDataSize += 1
# print("old size:", audioData.size)
# print("new size:", newDataSize)
# print("difference:", abs(audioData.size - newDataSize))
# print("seconds:", newDataSize / sampleRate)

segundos = newDataSize / sampleRate

while newDataSize % (sampleRate*SEGUNDOS_FILA) != 0:
    newDataSize += sampleRate

# print("new size:", newDataSize)
print("Longitud (s) de la cancion:", newDataSize / sampleRate)

audioData.resize(newDataSize)

#lo paso a matriz, tipo dataset
matrixAudioData = audioData.reshape( ( int(newDataSize/ (sampleRate*SEGUNDOS_FILA) ), sampleRate * SEGUNDOS_FILA) )
print("shape de la matriz de audio crudo:", matrixAudioData.shape)
# print(matrixAudioData[0])
# print(matrixAudioData[1])
print(matrixAudioData[1].shape)

######################################
# hago fft y lo junto a mi dataset   #
######################################
# from numpy.fft import fft, fftfreq
# CANT_FRECUENCIES = 2000
#
# print("haciendo fft a cada fila")
# i=0
# freqAmps = []
# for row in matrixAudioData:
#     freqs = fft(row, n = CANT_FRECUENCIES)
#     # freqAmps = np.vstack( (freqAmps, np.absolute (freqs)) )
#     freqAmps.append( np.absolute (freqs) )
#     # plt.figure(i)
#     # plt.plot(freqsAmp)
#     # i += 1
# # plt.show()
#
# freqAmps = np.array(freqAmps)
# # print("freqAmps", freqAmps.shape)
#
# matrixAudioData = np.hstack((matrixAudioData, freqAmps))
# print("matriz final con audio crudo y fft", matrixAudioData.shape)

############
# hago PCA #
############
print("")
from sklearn.decomposition import PCA
pca = PCA(n_components=200) #asi explicaba 95%
# pca = PCA(n_components=22050)
print("PCA...")
pca.fit(matrixAudioData)
print("Variance explained:", pca.explained_variance_ratio_.sum())
matrixAudioDataTransformed = pca.transform(matrixAudioData)
# print(matrixAudioDataTransformed[0])
print("shape transformed:", matrixAudioDataTransformed.shape)

###############
# hago kmeans #
###############
# print("")
# from sklearn import cluster
# kMeans = cluster.KMeans(n_clusters = 3, n_init=100, max_iter=3000, init="random")
# print("KMeans fitting...")
# kMeans.fit(matrixAudioDataTransformed)
# print("KMeans predicting...")
# labels = kMeans.predict(matrixAudioDataTransformed)
# print(labels.shape)
# print (labels)

#######################################
# hago cluster jerarquico con sklearn #
#######################################
# print("")
# from sklearn import cluster
# #lo mejor fue manhattan, average
# # aggloClustering = cluster.AgglomerativeClustering(affinity="manhattan", linkage="complete", compute_full_tree=True)
# aggloClustering = cluster.AgglomerativeClustering(affinity="manhattan", linkage="average", n_clusters = 2)
# print("Agglo fitting...")
# labels = aggloClustering.fit_predict(matrixAudioDataTransformed)
# #print(labels.shape)
# print (labels)
# print(aggloClustering.children_)
# minutos = 0
# segundos = 0
# for label in labels:
#     print(str(minutos) + ":" + str(segundos), str(label))
#     segundos += SEGUNDOS_FILA
#     if ( segundos >= 60 ):
#         minutos += 1
#         segundos -= 60






######################################
#  hago cluster jerarquico con scipy #
######################################
from scipy.cluster import hierarchy as h
from scipy.spatial import distance as dist
# import matplotlib.pyplot as plt

distanceMatrix = dist.pdist(matrixAudioDataTransformed, 'canberra') #canberra (0.8), cityblock (0.76), braycurtis(0.77)
clusters = h.linkage(distanceMatrix, 'average')


c,d=h.cophenet(clusters, distanceMatrix) #factor cofonético
print("cofonetica:",c)

THRESHOLD = 147.22

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

matrixAudioDataAndCluster = np.hstack((matrixAudioDataTransformed, clusters))
print(matrixAudioDataAndCluster)
print(matrixAudioDataAndCluster.shape)

plt.figure()
dn = h.dendrogram(clusters, color_threshold = THRESHOLD)
plt.show()




# print("")
# print("Plot...")
# import matplotlib.pyplot as plt
