#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 02:25:43 2017

@author: Shubhankar
"""

import tensorflow as tf

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score,mean_squared_error
# Example dummy data from Rendle 2010 
# http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
# Stolen from https://github.com/coreylynch/pyFM
# Categorical variables (Users, Movies, Last Rated) have been one-hot-encoded 
count=30000
x_data = np.random.rand(count,2)

y_data = np.array([(x[0]+2*x[1]+3+x[0]*4*x[1])+np.random.normal(0,1) for x in (list(x_data))])
y_data = np.array([(x[0]+2*x[1]+3)+np.random.normal(0,1) for x in (list(x_data))])

y_class =np.array([1.0 if ys>.5 else 0.0 for ys in y_data/np.max(y_data)])


# Let's add an axis to make tensoflow happy.
y_data.shape += (1, )
y_class.shape +=(1,)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(solver='saga',penalty='none')
clf.fit(x_data-np.mean(x_data,axis=0),y_class.ravel())
clf.score(x_data-np.mean(x_data,axis=0),y_class)
##################Linear#########################33
import tensorflow as tf
n, p = x_data.shape

# number of latent factors
k = 2

# design matrix
X = tf.placeholder('float', shape=[n, p])
# target vector
y = tf.placeholder('float', shape=[n, 1])

# bias and weights
w0 = tf.Variable(tf.random_normal([1],stddev = .01))
W = tf.Variable(tf.random_normal([p],stddev = .01))

# interaction factors, randomly initialized 
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

# estimate of y, initialized to 0.
y_hat = tf.Variable(tf.zeros([n, 1]))

linear_terms = tf.add(w0, tf.reduce_sum(tf.multiply(W,X),axis=1, keep_dims=True))

interactions = (tf.multiply(0.5,
                tf.reduce_sum(
                    tf.subtract(
                        tf.pow( tf.matmul(X, tf.transpose(V)), 2),
                        tf.matmul(tf.pow(X, 2), tf.transpose(tf.pow(V, 2)))),
                    1, keep_dims=True)))


l1_regularizer = tf.contrib.layers.l1_l2_regularizer(
   scale_l1=.005, scale_l2=.005, scope=None
)
weights = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)

y_hat = np.add(linear_terms,interactions)
mse = tf.reduce_mean(tf.square(tf.subtract(y, y_hat)))
loss = mse+regularization_penalty

eta = tf.constant(0.5)
optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

N_EPOCHS = 2000
# Launch the graph.
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(N_EPOCHS):
        indices = np.arange(n)
        np.random.shuffle(indices)
        x_data, y_data = x_data[indices], y_data[indices]
        sess.run(optimizer, feed_dict={X: x_data, y: y_data})

    print('MSE: ', sess.run(mse, feed_dict={X: x_data, y: y_data}))
#    print('Loss (regularized error):', sess.run(cost, feed_dict={X: x_data, y: y_data}))
    
    print('Predictions:', sess.run(y_hat, feed_dict={X: x_data, y: y_data}))
    print('Learnt weights:', sess.run(W, feed_dict={X: x_data, y: y_data}))
    print('Learnt weights:', sess.run(w0, feed_dict={X: x_data, y: y_data}))
    print('Learnt factors:', sess.run(V, feed_dict={X: x_data, y: y_data}))

#######################################classification
    


#############################################################
from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
#
#X_train = mnist.train.images
#Y_train = mnist.train.labels[:,0:2]
X_train = X_train.todense()
Y_train = y_train.astype('float')
X_test = X_test.todense()
Y_test = y_test.astype('float')
n, p = X_train.shape
classes = Y_train.shape[1]
k=100
# Parameters
learning_rate = 0.01
training_epochs = 4
batch_size = 1000
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, p]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, classes]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([p, classes]))
b = tf.Variable(tf.zeros([classes]))

# interaction factors, randomly initialized 
V = tf.Variable(tf.random_normal([k, p], stddev=0.01))

interactions = (tf.multiply(0.5,
                tf.reduce_sum(
                    tf.subtract(
                        tf.pow( tf.matmul(x, tf.transpose(V)), 2),
                        tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(V, 2)))),
                    1, keep_dims=True)))

l2_regularizer = tf.contrib.layers.l2_regularizer(
    scale=.005, scope=None
)
weights = tf.trainable_variables() # all vars of your graph
regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)


# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.09).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

predictions = 0
# Start training
sess = tf.Session()
sess.run(init)
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n/batch_size)
    # Loop over all batches
    for i in range(0,int(n/batch_size)):
        batch_xs = X_train[i*batch_size:(i+1)*batch_size]
        batch_ys = Y_train[i*batch_size:(i+1)*batch_size]
    # Loop over all batches
    
        # Fit training using batch data
        _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                      y: batch_ys})
        # Compute average loss
        avg_cost += c / total_batch
    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print ("Optimization Finished!")
#predictions = sess.run(pred,{x: X_train[0:100]})
# Test model
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy for 3000 examples
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print ("Accuracy test:", sess.run(accuracy,feed_dict = {x: X_test[0:4000], y: Y_test[0:4000]}))
print ("Accuracy train:", sess.run(accuracy,feed_dict = {x: X_train[0:4000], y: Y_train[0:4000]}))
