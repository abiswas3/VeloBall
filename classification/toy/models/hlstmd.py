from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import create_synthetic_data as datagen
import random
from IPython import embed
import sys
sys.path.append('../')

import newsgroups

LSTMCell = tf.nn.rnn_cell.BasicLSTMCell


class HLSTM(object):
    
    def __init__(self,
                 vocab_size,
                 max_sent_len,
                 max_doc_len,
                 num_classes,                 
                 hidden_state_dim=40,
                 batch_size=50,
                 embedding_dim=32,
                 learning_rate=0.1,
                 checkpoint_dir="checkpoint"):
        
        self.hidden_state_dim     = hidden_state_dim
        self.batch_size     = batch_size
        self.embedding_dim  = embedding_dim
        self.vocab_size     = vocab_size
        self.max_sent_len   = max_sent_len
        self.max_doc_len    = max_doc_len
        self.checkpoint_dir = checkpoint_dir
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate
        self.run()
        

    def build_model(self):
        self.input_data  = tf.placeholder(tf.int32, [self.batch_size, self.max_doc_len, self.max_sent_len])
        self.input_sent_lens  = tf.placeholder(tf.int32, [self.batch_size, self.max_doc_len])
        self.input_doc_lens  = tf.placeholder(tf.int32, [self.batch_size])
        self.output_data = tf.placeholder(tf.int32, [self.batch_size, self.num_classes])
        self.weights = tf.Variable(tf.random_normal([self.hidden_state_dim, self.num_classes]))
        self.biases = tf.Variable(tf.random_normal([self.num_classes]))
        
        with tf.device("/cpu:0"):
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size+1, self.embedding_dim], -1.0, 1.0))
            self.inputs = []
            for s in range(self.max_doc_len):
                self.inputs.append(tf.nn.embedding_lookup(self.embedding, self.input_data[:,s,:]))
                
        self.encode_word_cell = LSTMCell(self.hidden_state_dim)
        self.encode_sent_cell = LSTMCell(self.hidden_state_dim)

        word_state   = self.encode_word_cell.zero_state(self.batch_size, tf.float32)
        sent_encodings = []
        sent_state   = self.encode_sent_cell.zero_state(self.batch_size, tf.float32)

        print("IL",len(self.inputs))
        with tf.variable_scope("encode_RNN_word"):
            for s in range(self.max_doc_len):
                word_outputs, word_states = tf.nn.dynamic_rnn(cell=self.encode_word_cell,
                                                              dtype=tf.float32,
                                                              sequence_length=self.input_sent_lens[:,s],
                                                              inputs=self.inputs[s])
                
                sent_encodings.append(word_outputs[:,-1,:])

        with tf.variable_scope("encode_RNN_sentence"):
            sent_encodings = tf.stack(sent_encodings,axis=1)
            print("SES",sent_encodings.shape)
            doc_outputs, doc_states = tf.nn.dynamic_rnn(cell=self.encode_sent_cell,
                                                        dtype=tf.float32,
                                                        sequence_length=self.input_doc_lens,
                                                        inputs=sent_encodings)
        doc_encoding = doc_outputs[:,-1,:]

        self.result = tf.matmul(doc_encoding, self.weights) + self.biases
        print(doc_encoding.shape,self.result.shape)
        
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.result,
                                                                           labels=self.output_data))
                                   
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)
                                   
        self.correct_pred = tf.equal(tf.argmax(self.result,1), tf.argmax(self.output_data,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def get_batch(self, dataset, offset, size):
        
        return dataset['data'][offset:offset+size], dataset['targets'][offset:offset+size], dataset['doc_lens'][offset:offset+size], dataset['sent_lens'][offset:offset+size]
                                   
    def train(self, epochs=1000):
        
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        
            trainset = self.dataset
            offset = 0
            for step in range(epochs):                
                batch_x, batch_y, batch_d, batch_s = self.get_batch(trainset,offset,self.batch_size)
                batch_x, batch_y, batch_d, batch_s = np.array(batch_x), np.array(batch_y), np.array(batch_d), np.array(batch_s)
                                   
                sess.run(self.optimizer, feed_dict={self.input_data: batch_x,                                                    
                                                    self.output_data: batch_y,
                                                    self.input_sent_lens: batch_s,
                                                    self.input_doc_lens: batch_d})
                                   
                acc, loss = sess.run([self.accuracy,
                                      self.cost],
                                     feed_dict={self.input_data: batch_x,
                                                self.output_data: batch_y,
                                                self.input_sent_lens: batch_s,
                                                self.input_doc_lens: batch_d})
                                   
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                offset += self.batch_size
                if offset+self.batch_size > len(trainset['data']):
                    offset = random.randint(0,len(trainset['data'])-self.batch_size)
                #if step % save_iters == 0 and step != 0:
                #    self.save(sess)

            print("Optimization Finished!")
                                   

    
    def run(self):

        print("Building dataset")
        # Need to replace this with my function: docs is already good
        # need to make labels softmax
        # docs,labels = self.build_data(10)

        num_train = int(len(DATA)*0.8)
        self.dataset = {"data":DATA[:num_train,:,:],
                        "targets":LABELS[:num_train, :],
                        "doc_lens":DOC_LENS[:num_train],
                        "sent_lens":SENT_LENS[:num_train,:]}
        
        # Given it a test set as well
        # docs,labels = self.build_data(100)
        # self.testset = {"data":docs, "targets":labels}
        self.testset = {"data":DATA[num_train:,:,:], "targets":LABELS[num_train:, :]}
        print("Building model")
        self.build_model()
        print("Training")
        self.train(epochs=1000)


# vocab = newsgroups.vocab
# DATA = np.array(newsgroups.docs)
# LABELS = np.array(newsgroups.labels)
# DOC_LENS = np.array(newsgroups.doc_lens)
# SENT_LENS = np.array(newsgroups.sent_lens)

# print(DATA.shape, LABELS.shape, DOC_LENS.shape, SENT_LENS.shape)

# #embed()

# print("Vocab size: ", len(vocab))
# print("Data shape: ", DATA.shape)

# HLSTM(len(vocab), 25, 20)
if __name__ == '__main__':
    print('messi')                                   

                                   

