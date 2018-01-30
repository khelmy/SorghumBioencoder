import numpy as np
import tensorflow as tf
from . import NetDesign
from . import train
import argparse
import keras
from StringIO import StringIO
from tensorflow.python.lib.io import file_io

def train_model(train_file='../PickleDump/TrainData.npy',**args):
    losses = keras.losses
    #gs://sorghumencoder/SorghumBioencoder/PickleDump/TrainData.npy
    #(232302, 172) np array
    #trainData = np.load('../PickleDump/TrainData.npy')
    train_f = StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/TrainData.npy'))
    trainData = tf.Variable(initial_value=np.load(train_f), name='trainData')
    #(232302, 172) np array
    #testData = np.load('../PickleDump/TestData.npy')
    test_f = StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/TestData.npy'))
    testData = tf.Variable(initial_value=np.load(test_f), name='testData')
    #(232302,) np array
    #snpNodes = np.load('../PickleDump/SNPNodes.npy')
    nodes_f =  StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/SNPNodes.npy'))
    snpNodes = tf.Variable(initial_value=np.load(nodes_f), name='snpNodes')
    #(5685, 2) np array
    #How many (true) inputs does each downsample node have?
    #snpCounts = np.load('../PickleDump/SNPCounts.npy')
    counts_f = StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/SNPCounts.npy'))
    snpCounts = tf.Variable(initial_value=np.load(counts_f), name='snpCounts')

    #Load objects from .npy
    #sparseIndices = np.load('../PickleDump/SparseIndices.npy')
    indices_f = StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/SparseIndices.npy'))
    sparseIndices = tf.Variable(initial_value=np.load(indices_f), name='sparseIndices')
    #snpDenseShape = np.load('../PickleDump/SNPDenseShape.npy')
    dense_f = StringIO(file_io.read_file_to_string('gs://sorghumencoder/SorghumBioencoder/PickleDump/SNPDenseShape.npy'))
    snpDenseShape = tf.Variable(initial_value=np.load(dense_f), name='snpDenseShape')

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        trainData = trainData.eval(session=sess)
        testData = testData.eval(session=sess)
        snpNodes = snpNodes.eval(session=sess)
        snpCounts = snpCounts.eval(session=sess)
        sparseIndices = sparseIndices.eval(session=sess)
        snpDenseShape = snpDenseShape.eval(session=sess)

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

    def masked_cosine_proximity(y_true,y_pred):
        with tf.Session() as sess:
            y_pred_arr = y_pred.eval(session=sess)
            y_pred_arr_masked = y_pred_arr * out_mask_arr
            y_pred_masked = tf.convert_to_tensor(y_pred_arr_masked)
        return losses.cosine_proximity(y_true,y_pred_masked)

    model = NetDesign.buildModel(numSNPs=trainData[:,0].shape[0],
                        numLoci=snpCounts.shape[0],
                        maxLocus=maxLocus,
                        numEncode=256)
    model.compile(loss=masked_cosine_proximity,
                    optimizer='sgd')
    #model.save("../ModelDump/model_initial.h5")
    model.save('gs://sorghumencoder/SorghumBioencoder/ModelDump/model_initial.h5')
    model = train.train(trainData, model, sparseIndices, trainData)

    # To load model:
    # model = load_model('my_model.h5')

#From:
#https://github.com/liufuyang/kaggle-youtube-8m/blob/master/tf-learn/example-5-google-cloud/trainer/example5.py
if __name__ == '__main__':
    sess = tf.Session()
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
    sess.close()
