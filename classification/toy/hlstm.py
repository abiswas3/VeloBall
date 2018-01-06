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

BasicLSTMCell = tf.nn.rnn_cell.BasicLSTMCell


class HLSTM(object):
    def __init__(self,
                 vocab_size,
                 max_sent_len,
                 max_doc_len,                 
                 hidden_state_dim=16,
                 batch_size=50,
                 embedding_dim=16,
                 num_classes=3,
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
        
        inputs, outputs = datagen.generate_data(num_docs,
                                                max_sent=self.max_doc_len,
                                                min_sent=5,
                                                max_sent_len=self.max_sent_len,
                                                min_sent_len=5)

        tokens = {"A":1,"B":2,"C":3,"D":4,"E":5}        
        docs = []
        labels = []
        
        for i in range(len(inputs)):
            doc = []
            for sent in inputs[i].split("."):
                if len(sent) > 0: doc.append([tokens[c] for c in sent] + [0]*(self.max_sent_len - len(sent)))
            if len(doc) > 0: docs.append(doc + [[0]*self.max_sent_len]*(self.max_doc_len-len(doc)))
            labels.append([1,0] if outputs[0] == 1 else [0,1])


        # print(docs)
        # print(len(docs), len(docs[0]), len(docs[0][0]))
        print(labels)
        return docs,labels

    def get_batch(self, dataset, offset, size):
        return dataset['data'][offset:offset+size],dataset['targets'][offset:offset+size]
    
    def train(self,
              epochs=1000,
              save_iters=10):
        
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
            steps = len(self.testset['data'])//self.batch_size
            offset = 0
            for step in range(steps):
                batch_x, batch_y = self.get_batch(self.testset,offset,self.batch_size)
                batch_x, batch_y = np.array(batch_x), np.array(batch_y)
                acc, loss = sess.run([self.accuracy, self.cost], feed_dict={self.input_data: batch_x, self.output_data: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Test Accuracy= " + "{:.5f}".format(acc))
                offset += self.batch_size
                if offset+self.batch_size > len(self.testset['data']):
                    break
            # test_data,test_label = get_batch(testset,0,len(testset.data))
            # test_data = test_data.reshape((len(testset.data), timesteps, embedding_dim))
            # print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

    
    def run(self):

        print("Building dataset")
        # Need to replace this with my function: docs is already good
        # need to make labels softmax
        # docs,labels = self.build_data(10)

        num_train = int(len(DATA)*0.8)
        self.dataset = {"data":DATA[:num_train,:,:], "targets":output_class[:num_train, :]}
        
        # Given it a test set as well
        # docs,labels = self.build_data(100)
        # self.testset = {"data":docs, "targets":labels}
        self.testset = {"data":DATA[num_train:,:,:], "targets":output_class[num_train:, :]}        
        print("Building model")
        self.build_model()
        print("Training")
        self.train(epochs=1000)


if __name__ == '__main__':

    trainset = newsgroups.newsgroups_train
    testset = newsgroups.newsgroups_test
    vocab = {k:v+1 for k,v in zip(newsgroups.vocab, newsgroups.vocab.values())} # free up the 0 index
    rev_vocab = {v:k for k,v in zip(vocab, vocab.values())}
    vocab_size = len(vocab)
    print("Vocab size: ", len(vocab))

    ####################################
    # UNTIDY PREPROCESS TO BE CLEANED LATER
    ####################################
    news_articles = []
    for i in trainset['data']:
        temp = i.split('\n')[1:] # removing from information
        temp = [i.strip().split() for i in temp if len(i) > 0]
        news_articles.append(temp)
        
    docs = []
    for d in news_articles:
        doc = []
        for s in d:
            # minimum number of vocabulary words that has to be there in a sentence for it be included
            min_sen_len = max(len(s)//2, 1) 
            wcount=0
            for w in s:
                if w in vocab:
                    wcount +=1
                    
            # this sentence is not rubbish: use it
            if wcount > min_sen_len:
                doc.append(s)
            
        if len(doc) > 0:
            docs.append(doc) 
            
    news_articles = docs[:]

    MAX_DOC_LEN = 20 # throw away the rest :( 
    MIN_DOC_LEN = 10

    MAX_SEN_LEN = 15
    MIN_SEN_LEN = 6

    docs = []
    labels = []
    fucK_count = 0
    for index,doc in enumerate(news_articles):            
        
        # Document too small- don't care
        if len(doc) < MIN_DOC_LEN:
            continue
            
        new_doc = []    
        for num_sen, sent in enumerate(doc):
            
            # Sentence too small ignore
            if len(sent) < MIN_SEN_LEN:
                continue
                    
            nm_pad = MAX_SEN_LEN - len(sent)
            if nm_pad > 0:
                sent = [vocab[word] if word in vocab else 0 for word in sent] + [0 for i in range(nm_pad)]
            else:
                sent = [vocab[word] if word in vocab else 0 for word in sent[:MAX_SEN_LEN]]
                
            new_doc.append(sent)
            
        num_pad = 0 if len(new_doc) > MAX_DOC_LEN else MAX_DOC_LEN - len(new_doc)        
        # 0 pad those empty documents
        
        if num_pad > 0:
            for i in range(num_pad):
                new_doc.append([0 for k in range(MAX_SEN_LEN)])
                
        new_doc = np.array(new_doc[:MAX_DOC_LEN])    
        # small check
        if new_doc.shape[0] != MAX_DOC_LEN or new_doc.shape[1] != MAX_SEN_LEN :
            fucK_count += 1
            print("Old index: ", index, "New index: ", len(docs))
            print(new_doc.shape)
            print()
        
        docs.append(new_doc)    
        labels.append(trainset.target[index])

        
    news_articles = docs[:]
    labels = np.array(labels)

    ####################################
    # SHAPE INTO STYLE I NEED
    ####################################
    DATA = np.zeros((len(news_articles), MAX_DOC_LEN, MAX_SEN_LEN), dtype=int)
    for i in range(len(news_articles)):
        assert news_articles[i].shape[0] == MAX_DOC_LEN
        assert news_articles[i].shape[1] == MAX_SEN_LEN    
        DATA[i] = news_articles[i]

    output_class = np.zeros((DATA.shape[0], len(np.unique(labels))))
    for i in range(len(output_class)):
        output_class[i, labels[i]] = 1
        
    print(DATA.shape)
    print(output_class.shape)    
    
    HLSTM(vocab_size,
          MAX_SEN_LEN,
          MAX_DOC_LEN)
