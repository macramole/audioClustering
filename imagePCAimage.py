from scipy import misc
import numpy as np

IMAGE_WIDTH = 740
IMAGE_HEIGHT = 960

image = misc.imread("data/img/2.png")
print(image.shape)
#lo paso a matriz, tipo dataset
imageReshaped = image.reshape((100,-1))
print(imageReshaped.shape)

# pca_components = 5

#for pca_components in range(1,100):
for pca_components in range(1,100):
    #hago PCA
    print("")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components)
    print("PCA...")
    pca.fit(imageReshaped)
    print("Variance explained:", pca.explained_variance_ratio_.sum())
    imageTransformed = pca.transform(imageReshaped)
    # print(matrixAudioDataTransformed[0])
    print("shape transformed:", imageTransformed.shape)
    # misc.imsave("1_pca.png", imageTransformed)
    # exit()

    #como se ve?
    print("")
    print("Cómo se ve?")
    imageInverse = pca.inverse_transform(imageTransformed)
    print("shape inverse transform:", imageInverse.shape)
    imageInverseColor = imageInverse.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    misc.imsave("1_pca" + str(pca_components) + "_" + str(int(pca.explained_variance_ratio_.sum()*100)) + ".png", imageInverseColor)

# newAudioData = matrixAudioDataInverseTransform.reshape(-1)
# newAudioData = newAudioData.astype(np.int16, copy=False)
# print("grabando nuevo audio...")
# wavfile.write("muchachita_new.wav", sampleRate, newAudioData)
# print("listo")
