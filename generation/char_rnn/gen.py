import tensorflow as tf
import numpy as np
import os, random
from IPython import embed

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell

from chat import *

class CharRNN:
    def __init__(self,
                 maxlen,
                 hidden_state_dim=50,
                 batch_size=10,
                 learning_rate=0.001,
                 checkpoint_dir="checkpoint"):
        self.maxlen = maxlen
        self.hidden_state_dim = hidden_state_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        #self.chat = chat("short", self.maxlen)
        self.chat = chat("names/English.txt", self.maxlen)
        self.run()

    def build_model(self):
        self.input_data = tf.placeholder(tf.float32, [self.batch_size, self.maxlen, DIM])
        self.input_lens = tf.placeholder(tf.float32, [self.batch_size])
        self.output_data = tf.placeholder(tf.float32, [self.batch_size, self.maxlen, DIM])
        self.weights = tf.Variable(tf.random_normal([self.hidden_state_dim, DIM]))
        self.biases = tf.Variable(tf.random_normal([DIM]))
        self.cell = BasicLSTMCell(self.hidden_state_dim)

        char_state = self.cell.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope("RNN"):
            self.char_output, char_state = tf.nn.dynamic_rnn(cell=self.cell, dtype=tf.float32, sequence_length=self.input_lens, inputs=self.input_data)
        # char_output is [batchsize, maxlen, hiddendim] i.e. [[[b0s0c0,b0s0c1,...],...[b0sSc0,...]],...]
        # flat_char_output is [batchsize*maxlen, hiddendim]
        # result is [batchsize*maxlen, embeddingdim]
        flat_char_output = tf.reshape(self.char_output, [self.batch_size*self.maxlen, self.hidden_state_dim])
        print("FO",flat_char_output.shape)
        result = tf.matmul(flat_char_output, self.weights) + self.biases
        print("RS",result.shape)
        flat_outputs = tf.reshape(self.output_data, [self.batch_size*self.maxlen, DIM])
        self.outputs = tf.reshape(result,[self.batch_size, self.maxlen, DIM])
        print("FS",flat_outputs.shape)
        self.cost = tf.nn.softmax_cross_entropy_with_logits(logits=result, labels=flat_outputs)
        print("CS",self.cost.shape)
        self.loss = tf.reshape(self.cost, [self.batch_size, self.maxlen])
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.correct_pred = tf.equal(tf.argmax(flat_char_output,1), tf.argmax(flat_outputs,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.line_loss = tf.reduce_mean(self.loss, 1)
        self.batch_loss = tf.reduce_mean(self.line_loss)
        
    def train_model(self, epochs=100000):
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        
            offset = 0
            for step in range(epochs):
                batch_x, batch_l, batch_y = self.chat.get_batch(self.batch_size,offset)
                print(batch_l[0])
                print(self.chat.to_string(batch_x[0]))
                print(self.chat.to_string(batch_y[0]))
                batch_x, batch_l, batch_y = np.array(batch_x), np.array(batch_l), np.array(batch_y)
                print(batch_x.shape,batch_l.shape,batch_y.shape)
                sess.run(self.optimizer, feed_dict={self.input_data: batch_x,
                                                    self.output_data: batch_y,
                                                    self.input_lens: batch_l})
                acc, loss, batch_loss, out = sess.run([self.accuracy, self.loss, self.batch_loss, self.outputs], feed_dict={self.input_data: batch_x,
                                                                            self.output_data: batch_y,
                                                                            self.input_lens: batch_l})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(batch_loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                print("Guess",self.chat.to_string(out[0]))
                offset += self.batch_size
                if offset+self.batch_size > len(self.chat.data):
                    offset = random.randint(0,len(self.chat.data)-self.batch_size)

            print("Optimization Finished!")

    def run(self):
        self.build_model()
        self.train_model()

CharRNN(10,batch_size=12)
