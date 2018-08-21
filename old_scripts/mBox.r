library(biotools)
df = read.csv("/media/macramole/GARBER/matrixAudioData.tsv", sep = "\t", header = F)
clusters = read.csv("~/Code/python/audioClustering/audioClusteringResult.tsv", sep="\t", header = T)
mbox = boxM( df, clusters$cluster )
