from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import create_synthetic_data as datagen
import random
from IPython import embed

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell


class HLSTM(object):
    def __init__(self, vocab, max_sent_len, max_doc_len, hidden_state_dim=10, batch_size=2, embedding_dim=6, num_classes=2, learning_rate=0.01, checkpoint_dir="checkpoint"):
        self.hidden_state_dim     = hidden_state_dim
        self.batch_size     = batch_size
        self.embedding_dim  = embedding_dim
        self.vocab          = vocab
        self.vocab_size     = len(vocab)+1
        self.max_sent_len   = max_sent_len
        self.max_doc_len    = max_doc_len
        self.checkpoint_dir = checkpoint_dir
        self.num_classes    = num_classes
        self.learning_rate  = learning_rate
        self.run()

    def build_model(self):
        self.input_data  = tf.placeholder(tf.int32, [self.batch_size, self.max_doc_len, self.max_sent_len])
        self.output_data = tf.placeholder(tf.int32, [self.batch_size, self.num_classes])
        self.weights = tf.Variable(tf.random_normal([self.hidden_state_dim, self.num_classes]))
        self.biases = tf.Variable(tf.random_normal([self.num_classes]))
        
        with tf.device("/cpu:0"):
            self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_dim], -1.0, 1.0))
            self.inputs = []
            for s in range(self.max_doc_len):
                self.inputs.append(tf.nn.embedding_lookup(self.embedding, self.input_data[:,s,:]))
                
        self.encode_word_cell = BasicLSTMCell(self.hidden_state_dim)
        self.encode_sent_cell = BasicLSTMCell(self.hidden_state_dim)

        word_state   = self.encode_word_cell.zero_state(self.batch_size, tf.float32)
        sent_encodings = []
        sent_state   = self.encode_sent_cell.zero_state(self.batch_size, tf.float32)
        with tf.variable_scope("encode_RNN_word"):
            for s in range(self.max_doc_len):
                for t in range(self.max_sent_len):
                    if not (t == 0 and s == 0): tf.get_variable_scope().reuse_variables()
                    (cell_output, word_state) = self.encode_word_cell(self.inputs[s][:,t,:], word_state)
                sent_encodings.append(cell_output)

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

    def save(self, sess):
        self.saver = tf.train.Saver()
        print("Saving checkpoints...")
        model_dir, model_name = "./model","mymodel"
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(sess, os.path.join(checkpoint_dir, model_name))

    def load(self, sess, checkpoint_dir):
        model_dir, model_name = self.get_model_dir()
        checkpoint_dir = os.path.join(self.checkpoint_dir, model_dir)
        self.saver = tf.train.Saver()

        print("Loading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        return False 

    def build_data(self, num_docs):
        inputs,outputs = datagen.generate_data(100,max_sent=self.max_doc_len, max_sent_len=self.max_sent_len)
        tokens = {"A":1,"B":2,"C":3,"D":4,"E":5}
        
        docs = []
        labels = []
        
        for i in range(len(inputs)):
            doc = []
            for sent in inputs[i].split("."):
                if len(sent) > 0: doc.append([tokens[c] for c in sent] + [0]*(self.max_sent_len - len(sent)))
            if len(doc) > 0: docs.append(doc + [[0]*self.max_sent_len]*(self.max_doc_len-len(doc)))
            labels.append([1,0] if outputs[0] == 1 else [0,1])

        self.dataset = {"data":docs, "targets":labels}
        return docs,labels

    def get_batch(self, dataset, offset, size):
        return dataset['data'][offset:offset+size],dataset['targets'][offset:offset+size]
    
    def train(self, epochs=100, save_iters=10):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        
            trainset = self.dataset
            offset = 0
            for step in range(epochs):
                batch_x, batch_y = self.get_batch(trainset,offset,self.batch_size)
                batch_x, batch_y = np.array(batch_x), np.array(batch_y)
                # batch_x = ((doc0_word0,doc0_word1,doc0_word2,...),...,(docB_word0,docB_word1,docB_word2,...))
                # batch_y = ((doc0_is_class0,,doc0_is_class1,doc0_is_class2),...,(docB_is_class0,,docB_is_class1,docB_is_class2))
                # batch_s = (doc0_len,...,docB_len)
                #print("X",batch_x[-1],batch_x[0].shape,batch_x.shape)
                #print("Y",batch_y[-1],batch_y[0].shape,batch_y.shape)
                sess.run(self.optimizer, feed_dict={self.input_data: batch_x, self.output_data: batch_y})
                acc, loss = sess.run([self.accuracy, self.cost], feed_dict={self.input_data: batch_x, self.output_data: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
                offset += self.batch_size
                if offset+self.batch_size > len(trainset['data']):
                    offset = random.randint(0,len(trainset['data'])-self.batch_size)
                #if step % save_iters == 0 and step != 0:
                #    self.save(sess)

            print("Optimization Finished!")
            # test_data,test_label = get_batch(testset,0,len(testset.data))
            # test_data = test_data.reshape((len(testset.data), timesteps, embedding_dim))
            # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    
    def run(self):
        print("Building dataset")
        self.build_data(100)
        print("Building model")
        self.build_model()
        print("Training")
        self.train()

HLSTM(["A","B","C","D","E"],20,10)
