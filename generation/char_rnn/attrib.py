import tensorflow as tf
import numpy as np
import os, random
from IPython import embed

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell

class UnlabelledFileLineCharDataset:
    def __init__(self, filename, maxlen):
        with open(filename,"rb") as f:
            d = f.read()
        self.data = []
        datum = []
        for i in range(len(d)):
            datum.append(self.onehot(int(d[i])))
            if c == chr('\n'):
                if len(datum) <= maxlen: # if the line meets our maxlen requirement, pad and add
                    self.data.append(datum + [self.pad() for j in range(maxlen-len(datum))])
                datum = [] # start a new line regardless
    def pad(self):
        return self.onehot(257)
    def onehot(self, byte):
        return [1 if byte == i else 0 for i in range(258)]
    def get_batch(self, offset, batch_size):
        return np.array(self.data[offset:offset+batch_size])

class CharRNN:
    def __init__(self,
                 max_len,
                 hidden_state_dim=128,
                 batch_size=10,
                 learning_rate=0.01,
                 checkpoint_dir="checkpoint"):
        self.maxlen = maxlen
        self.hidden_state_dim = hidden_state_dim
        self.batch_size = batch_size
        self.embedding_dim = 258
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.run()

    def build_model(self):
        self.input_data = tf.placeholder(tf.int32, [self.batch_size, self.maxlen, self.embedding_dim])
        self.weights = tf.Variable(tf.random_normal([self.hidden_state_dim, self.embedding_dim]))
        self.biases = tf.Variable(tf.random_normal([self.embedding_dim]))
        
        self.cell = BasicLSTMCell(self.hidden_state_dim)

        char_state   = self.encode_word_cell.zero_state(self.batch_size, tf.float32)
        pred = []
        with tf.variable_scope("RNN"):
            for s in range(self.maxlen):
                if s > 0: tf.get_variable_scope().reuse_variables()
                (char_output, char_state) = self.cell(self.input_data[s][:,t,:], word_state)
                pred.append(cell_output)

        with tf.variable_scope("encode_RNN_sentence"):
            for s in range(self.max_doc_len):
                if s > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, sent_state) = self.encode_sent_cell(sent_encodings[s],
                                                                  sent_state)
            doc_encoding = cell_output

        self.result = tf.matmul(doc_encoding, self.weights) + self.biases
        print(doc_encoding.shape,self.result.shape)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.result, labels=self.output_data))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.correct_pred = tf.equal(tf.argmax(self.result,1), tf.argmax(self.output_data,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

