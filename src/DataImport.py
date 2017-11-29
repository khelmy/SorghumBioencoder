import numpy as np
import tensorflow as tf

testData = np.genfromtxt('../SNPData/TestData.csv',delimiter=',')
trainData = np.genfromtxt('../SNPData/TrainData.csv',delimiter=',')
snpNodes = np.genfromtxt('../SNPData/SNPNodes.csv',delimiter=',')
snpCounts = np.genfromtxt('../SNPData/SNPCounts.csv',delimiter=',')

#So we can sparsify them later
#negative = 1, positive = 2, no record = 0
testData = testData + 1
trainData = trainData + 1

np.save('../PickleDump/TrainData.npy',trainData)
np.save('../PickleDump/TestData.npy',testData)
np.save('../PickleDump/SNPNodes.npy',snpNodes)
np.save('../PickleDump/SNPCounts.npy',snpCounts)

"""
trainData = np.load('../PickleDump/TrainData.npy')
testData = np.load('../PickleDump/TestData.npy')
snpNodes = np.load('../PickleDump/SNPNodes.npy')
snpCounts = np.load('../PickleDump/SNPCounts.npy')
"""

print(trainData.shape)
print(testData.shape)
print(snpNodes.shape)
print(snpCounts.shape)
