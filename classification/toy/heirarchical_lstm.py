import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import sys, re, random
from IPython import embed

learning_rate = 0.001
batch_size = 1
display_step = 200

timesteps = 20
num_hidden = 128
num_classes = 2
epochs = 20

trainset = {"data":["ABDDEEDD.ACDEDEDEDE.","ABAC.EACAB.","ACAAAE.DDABDDEE.","ABACABEEE."],
            "target":[[0,1],[0,1],[1,0],[1,0]]}
testset = {"data":["ABA.AEEEAC.","BCDDACEEAB.","AAABDD.EEEEE.","ACDDDDAB.ABABEEE."],
           "target":[[0,1],[0,1],[1,0],[1,0]]}
tokens = {"A":0,"B":1,"C":2,"D":3,"E":4,".":5}
num_tokens = 6
word_embedding_dim = num_tokens # since we use one-hot encoding
max_sentence_len = 50
max_doc_len = 50

# Takes doc and returns a matrix like [[s0w0,s0w1,...],[s1w0,s1w1,...],...]
# where siwj is the one-hot embedding of word j in sentence i
def split_and_tokenise_doc(raw_doc):
    doc = []
    sentence = []
    for i in range(len(raw_doc)):
        sentence.append([1 if tokens[doc[i]] == j else 0 for j in range(num_tokens)])
        if doc[i] == ".":
            doc.append(sentence+[[0 for j in range(num_tokens)] for k in range(max_sentence_len-len(sentence))])
            sentence = []
    return doc

x = tf.placeholder(tf.int64, shape=[batch_size, max_document_len, max_sentence_len, word_embedding_dim])
y = tf.placeholder(tf.int32, [batch_size, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

num_layers = 2
layers = [BasicLSTMCell(num_hidden) for i in range(num_layers)]





# Outputs a tensor of size [batch_size, output_dim] I think?
def LSTM(input_x, seqlens, hidden_dim, output_dim):
    print("I",input_x)
    print("S",seqlens)
    lstm_cell = rnn.BasicLSTMCell(hidden_dim)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_x, sequence_length=seqlens, dtype=tf.float32)
    return tf.matmul(outputs[-1], tf.Variable(tf.random_normal([hidden_dim, output_dim]))) + tf.Variable(tf.random_normal([output_dim]))

print("Creating network")

# Create a sequence of max_document_len LSTMs, each with hidden dimension 100
# Each of these will generate a 100-dimensional output representing the sentence's embedding
# This will feed the ith sentence of every document in the batch into the ith LSTM 
sentence_outputs = [LSTM(x[0:-1][i], sentence_lens[0:-1][i], 100, 100) for i in range(max_document_len)]

# Will feed the results of the above LSTMs into a further LSTM with hidden dimension 100 and output dimension 2
document_output = LSTM(sentence_outputs, document_lens, 100, num_classes)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=document_output, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
