import sys
import argparse
import os
import utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("audioFilesPath", help="Path to audio files to parse")
parser.add_argument("-ft", "--fileType", default = "wav", help="File type of audio files. Default: wav")
parser.add_argument("-f", "--features", default = "mfcc", help="Features to extract from audio. stft or mfcc. Default: mfcc")
parser.add_argument("-n", "--qtyFilesToProcess", default = "-1", help="Quantity of files to process. Default: all files found")
parser.add_argument("-t", "--hierarchicalClusteringThreshold", default = 0.995, help="Threshold for hierarchical clustering. Default: 0.995", type = float)


args = parser.parse_args()

if os.path.isdir(args.audioFilesPath):
    print("reading directory", args.audioFilesPath, "...")
else:
    print("audioFilesPath not found :(")
    exit()

audioFiles = utils.findMusic( args.audioFilesPath, args.fileType )
print(len(audioFiles), args.fileType, "files found")

if len(audioFiles) == 0:
    exit()

args.qtyFilesToProcess = int(args.qtyFilesToProcess)
if args.qtyFilesToProcess == -1:
    args.qtyFilesToProcess = len(audioFiles)


print("now processing")

audioData = utils.getAudioData( audioFiles, features = args.features, qtyFilesToProcess = args.qtyFilesToProcess )
print("DONE!")

print("")
print("Now doing PCA")
audioDataTransformed = utils.doPCA(audioData)


print("")
print("Now doing hierarchical clustering")
clusters = utils.doHierachicalClustering( audioDataTransformed, args.hierarchicalClusteringThreshold )
print("Clusters found:", np.unique(clusters).size)

print("Now doing t-SNE")
tsne = utils.doTSNE( audioDataTransformed )
print("DONE!")

print("")
print("Now saving session file")

import json

jsonSession = {
    "audioFilesPath" : args.audioFilesPath,
    "tsv" : ""
}

audioFilesForExport = list( map( lambda x : x[len(args.audioFilesPath):], audioFiles ) )
output = np.c_[ tsne, clusters, audioFilesForExport ]

tsv = ""
for row in output:
    for field in row:
        tsv += field + "\t"
    # tsv = tsv[0:-1]
    tsv += "\n"

jsonSession["tsv"] = tsv

with open('session.json', 'w') as fp:
    json.dump(jsonSession, fp)
