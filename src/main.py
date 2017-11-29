import numpy as np
import tensorflow as tf
from . import NetDesign
from . import train
import argparse

losses = tf.keras.losses

sess = tf.Session()

#(232302, 172) np array
trainData = np.load('./PickleDump/TrainData.npy')
#(232302, 172) np array
testData = np.load('./PickleDump/TestData.npy')
#(232302,) np array
snpNodes = np.load('./PickleDump/SNPNodes.npy')
#(5685, 2) np array
#How many (true) inputs does each downsample node have?
snpCounts = np.load('./PickleDump/SNPCounts.npy')

#Load objects from .npy
sparseIndices = np.load('./PickleDump/SparseIndices.npy')
snpDenseShape = np.load('./PickleDump/SNPDenseShape.npy')

def train_model(train_file='./PickleDump/TrainData.npy',**args):
    maxLocus = snpCounts[:,1].max()
    maskIndices = sparseIndices[:,1]

    #Example SparseTensor object. Just replace values to represent SNP sequences
    out_mask = tf.SparseTensor(indices=sparseIndices,
                                values=np.ones((sparseIndices.shape[0],)),
                                dense_shape=snpDenseShape)
    out_mask_arr = tf.sparse_tensor_to_dense(out_mask).eval(session=sess)

    #in_val = tf.SparseTensor(indices=main.sparseIndices,
    #                            values=main.trainData[:,0],
    #                            dense_shape=main.snpDenseShape)
    #in_val_arr = tf.sparse_tensor_to_dense(in_val).eval(session=sess)

    #Build the model (256 is arbitrary)

    def masked_hinge_loss(y_true,y_pred):
        y_pred_arr = y_pred.eval(session=sess)
        y_pred_arr_masked = y_pred_arr * out_mask_arr
        y_pred_masked = tf.convert_to_tensor(y_pred_arr_masked)
        return losses.hinge(y_true,y_pred_masked)

    model = NetDesign.buildModel(numSNPs=trainData[:,0].shape[0],
                        numLoci=snpCounts.shape[0],
                        maxLocus=maxLocus,
                        numEncode=256)
    model.compile(loss=masked_hinge_loss,
                    optimizer='sgd')
    model.save("./ModelDump/model_initial.h5")
    model = train.train(trainData, model, sparseIndices,trainData,sess)

    # To load model:
    # model = load_model('my_model.h5')
    sess.close()

#From:
#https://github.com/liufuyang/kaggle-youtube-8m/blob/master/tf-learn/example-5-google-cloud/trainer/example5.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)
