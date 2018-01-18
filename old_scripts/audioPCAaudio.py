from scipy.io import wavfile
import numpy as np

# audioFile = wavfile.read("audio_44100_16.wav");
# audioFile = wavfile.read("audio_22050_16.wav");
# audioFile = wavfile.read("audio_22050_8.wav");
# audioFile = wavfile.read("audio_22050_8_mono.wav");
sampleRate, audioData = wavfile.read("muchachita_22050_16_mono.wav")
# sampleRate, audioData = wavfile.read("audio_22050_16_mono.wav")

#trunco para que quede justo al cortar de a 1 segundo
newDataSize = audioData.size
while newDataSize % sampleRate != 0:
    newDataSize -= 1
print("old size:", audioData.size)
print("new size:", newDataSize)
print("difference:", audioData.size - newDataSize)
audioData.resize(newDataSize)

#lo paso a matriz, tipo dataset
matrixAudioData = audioData.reshape( ( int(newDataSize/sampleRate), sampleRate) )
print("matrix shape:", matrixAudioData.shape)
print(matrixAudioData[0])
# print(matrixAudioData[1])
# print(matrixAudioData[1].size)

#hago PCA
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


#como suena?
print("")
print("Cómo suena este monstruo?")
matrixAudioDataInverseTransform = pca.inverse_transform(matrixAudioDataTransformed)
print("shape inverse transform:", matrixAudioDataInverseTransform.shape)
print(matrixAudioDataInverseTransform[0])

newAudioData = matrixAudioDataInverseTransform.reshape(-1)
newAudioData = newAudioData.astype(np.int16, copy=False)
print("grabando nuevo audio...")
wavfile.write("muchachita_new.wav", sampleRate, newAudioData)
print("listo")
