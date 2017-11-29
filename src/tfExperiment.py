import numpy as np
import tensorflow as tf
import processing as pr

sess = tf.Session()

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

print(node1, node2)

node3 = node1 + node2
print("node3:", node3)
print("sess.run(node3):", sess.run(node3, {node1:[1,2,3], node2:[2,3,4]}))

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.initialize_all_variables()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

#Load objects from .npy
sparseIndices = np.load('../PickleDump/SparseIndices.npy')
snpDenseShape = np.load('../PickleDump/SNPDenseShape.npy')

#Example SparseTensor object. Just replace values to represent SNP sequences
tf.SparseTensor(indices=sparseIndices,
                values=np.ones((sparseIndices.shape[0],)),
                dense_shape=snpDenseShape)
