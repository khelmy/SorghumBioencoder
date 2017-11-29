import numpy as np
import tensorflow as tf

trainData = np.load('../PickleDump/TrainData.npy')
testData = np.load('../PickleDump/TestData.npy')
snpNodes = np.load('../PickleDump/SNPNodes.npy')
snpCounts = np.load('../PickleDump/SNPCounts.npy')

#Each neuron accepts the same amount as the largest node
nodeInputSize = snpCounts[:,1].max()
sparseIndices = np.array([])
snpIndex = 0
for i in range(snpCounts.shape[0]):
    rangeSize = snpCounts[i,1]
    denseRange = np.arange(snpIndex, snpIndex + rangeSize)
    sparseIndices = np.concatenate([sparseIndices, denseRange])
    snpIndex += nodeInputSize
sparseIndices = np.stack([np.zeros(sparseIndices.shape),sparseIndices],-1)
snpDenseShape = np.array([1, snpCounts.shape[0] * nodeInputSize])

print(sparseIndices)
print(snpDenseShape)

np.save('../PickleDump/SparseIndices.npy',sparseIndices)
np.save('../PickleDump/SNPDenseShape.npy',snpDenseShape)
