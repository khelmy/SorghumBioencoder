import numpy as np
import tensorflow as tf

#How to access one SNP seq
#main.trainData[:,0]
def train(trainData, model, sparseIndices, sess):
    for i in range(trainData.shape[1]):
        snpTrainI = tf.SparseTensor(indices=sparseIndices,
                                    values=trainData[:,i],
                                    dense_shape=snpDenseShape)
        snpTrainDenseI = tf.sparse_tensor_to_dense(snpTrainI).eval(session=sess)
        train_loss_i = model.train_on_batch(snpTrainDenseI, snpTrainDenseI)
        print(train_loss_i)
        # DEBUGGING
        #if i % 5 == 0:
            #model.save("./ModelDump/model_iter.h5")
            #model.save('gs://sorghumencoder/SorghumBioencoder/ModelDump/model_iter.h5')
    #model.save("./ModelDump/model_final.h5")
    model.save('gs://sorghumencoder/SorghumBioencoder/ModelDump/model_final.h5')
    return model
