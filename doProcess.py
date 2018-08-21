#!/usr/bin/env python
import sys
import argparse
import os
import utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("audioFilesPath", help="Path to audio files to parse")
parser.add_argument("-ft", "--fileType", default = "wav", help="File type of audio files. Default: wav")
parser.add_argument("-f", "--features", default = "stft", help="Features to extract from audio. stft or mfcc. Default: stft")
parser.add_argument("-n", "--qtyFilesToProcess", default = "-1", help="Quantity of files to process. Default: all files found")
parser.add_argument("-t", "--hierarchicalClusteringThreshold", default = 0.995, help="Threshold for hierarchical clustering. Default: 0.995", type = float)
parser.add_argument("-am", "--saveAudioMatrix", default = True, help="Saves numpy file to reuse. Default: True", type = bool)


args = parser.parse_args()

if os.path.isdir(args.audioFilesPath):
    if args.audioFilesPath[-1] != "/":
        args.audioFilesPath += "/"
    print("reading directory", args.audioFilesPath, "...")
else:
    print("audioFilesPath not found :(")
    exit()

audioFilesPathDirectoryName = os.path.basename(args.audioFilesPath[0:-1])

audioFiles = utils.findMusic( args.audioFilesPath, args.fileType )
print(len(audioFiles), args.fileType, "files found")

if len(audioFiles) == 0:
    exit()

args.qtyFilesToProcess = int(args.qtyFilesToProcess)
if args.qtyFilesToProcess == -1:
    args.qtyFilesToProcess = len(audioFiles)

if not os.path.isfile( os.path.join(args.audioFilesPath, "../", '%s.%s.npy' % (audioFilesPathDirectoryName, args.features) ) ):
    print("now processing")
    audioData = utils.getAudioData( audioFiles, features = args.features, qtyFilesToProcess = args.qtyFilesToProcess )
    if args.saveAudioMatrix:
        np.save( os.path.join(args.audioFilesPath, "../", '%s.%s' % (audioFilesPathDirectoryName, args.features) ) , audioData)
        print("Audio matrix saved!")
else:
    print("Using %s" % os.path.join(args.audioFilesPath, "../", '%s.%s.npy' % (audioFilesPathDirectoryName, args.features) ) )
    audioData = np.load( os.path.join(args.audioFilesPath, "../", '%s.%s.npy' % (audioFilesPathDirectoryName, args.features) ) )

print("DONE!")

print("")
print("Now doing PCA")
audioDataTransformed = utils.doPCA(audioData)


print("")
print("Now doing hierarchical clustering")
print("Threshold is:", args.hierarchicalClusteringThreshold)
clusters = utils.doHierachicalClustering( audioDataTransformed, args.hierarchicalClusteringThreshold )
print("Clusters found:", np.unique(clusters).size)

print("")
print("Now doing t-SNE")
tsne = utils.doTSNE( audioDataTransformed, 2 )
print("DONE!")

print("")
print("Now doing DBScan")
clusters = utils.doDBScan( tsne )
print("Clusters found:", np.unique(clusters).size)
print("DONE!")

print("")
print("Now saving session file")

import json

jsonSession = {
    "audioFilesPath" : "%s/" % audioFilesPathDirectoryName,
    "tsv" : ""
}

audioFilesForExport = list( map( lambda x : x[len(args.audioFilesPath):], audioFiles ) )
output = np.c_[ tsne, np.repeat(0, tsne.shape[0] ) ,clusters, audioFilesForExport ]

tsv = ""
for row in output:
    for field in row:
        tsv += field + ","
    tsv = tsv[0:-1]
    tsv += "|"

jsonSession["tsv"] = tsv

with open( os.path.join(args.audioFilesPath, "../", '%s.%s.json' % ( audioFilesPathDirectoryName, args.features )  ), 'w') as fp:
    json.dump(jsonSession, fp)

print("All done, have a great day")
