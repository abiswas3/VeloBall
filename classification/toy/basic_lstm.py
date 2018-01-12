""" Recurrent Neural Network.

A ghetto LSTM hacked together for sanity check
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from make_training_data import get_data

import sys

DATA, LABELS, vocab, rvocab, test_data, test_labels = get_data()
embedding_dim = 12

print(DATA.shape, LABELS.shape)
#embed()
print("Vocab size: ", len(vocab))
print("Data shape: ", DATA.shape)



# Training Parameters
learning_rate = 0.1
training_steps = 10000
batch_size = 50
display_step = 200

# Network Parameters
timesteps = 32 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 8 # MNIST total classes (0-9 digits)

# tf Graph input
# variable batch size
# by ma sent length
X = tf.placeholder(tf.int32, [None, 
                              timesteps])

Y = tf.placeholder("float", [None, num_classes])
embedding = tf.Variable(tf.random_uniform([len(vocab)+1,
                                           embedding_dim], -1.0, 1.0))

inputs = tf.nn.embedding_lookup(embedding, X)
# X = tf.placeholder("float", [None, DATA.shape[1], num_input])


# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(inputs, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        choices = [i for i in range(len(DATA))]
        inds = np.random.choice(choices, batch_size, replace=False)
        batch_x = DATA[inds]
        batch_y = LABELS[inds]
                                
        
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # # Reshape data to get 28 seq of 28 elements
        # batch_x = batch_x.reshape((batch_size, timesteps, num_input))
                
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print('Train stats')
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            print("Test : STATS")
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: test_data,
                                                                 Y: test_labels})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Testing  Accuracy= " + \
                  "{:.3f}".format(acc))

    
            print()
