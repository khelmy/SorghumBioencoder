import numpy as np
import tensorflow as tf

Dense = tf.keras.layers.Dense
LocallyConnected1D = tf.keras.layers.LocallyConnected1D
Activation = tf.keras.layers.Activation
Sequential = tf.keras.models.Sequential

#numSNPs = 232302
#numLoci = 5685
#numEncode = 256
def buildModel(numSNPs, numLoci, maxLocus, numEncode):
    #First, change size from numSNPs to numLoci * maxLocus (pad)
    #Debugging
    print('Building Model')
    #print(type(numSNPs))
    #print(type(numLoci))
    #print(type(maxLocus))
    #print(type(numEncode))
    maxLocus = int(maxLocus)
    #print(maxLocus)
    #print(numLoci * maxLocus)
    model = Sequential([
        #Mask
        #Masking(mask_value=0, input_shape=(1,numLoci * maxLocus)),
        #Downsample layer
        #LocallyConnected2D(numLoci, (1,maxLocus)),

        #Downsample layer (new)
        LocallyConnected1D(numLoci, (maxLocus,), input_shape=(numLoci * maxLocus,1)),

        #This is equivalent to an encoding layer.
        Dense(numEncode),       #Part of encoding layer (bring down to 256)
        Activation('sigmoid'),  #Part of encoding layer (sigmoid)
        Dense(numLoci),         #Part of encoding layer (bring up from 256)
        #End encoding layer

        Dense(numLoci * maxLocus) #Upsample layer
    ])
    #Remember to postfilter after!
    return model
