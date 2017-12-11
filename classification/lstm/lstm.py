import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import sys, re, random
sys.path.append("../")
import newsgroups
from IPython import embed

learning_rate = 0.001
batch_size = 256
display_step = 200

timesteps = 100
num_hidden = 128
num_classes = 3
epochs = 20

trainset = newsgroups.newsgroups_train
testset = newsgroups.newsgroups_test
vocab = newsgroups.vocab
vocab_size = len(vocab)
embedding_dim = 128
embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))

x = tf.placeholder(tf.int64, shape=[batch_size])
y = tf.placeholder(tf.int32, [None, num_classes])
dropout_keep_prob = tf.placeholder(tf.float32)

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def LSTM(input_x):
    print("I",input_x)
    lstm_cell = rnn.BasicLSTMCell(num_hidden)
    outputs, states = rnn.static_rnn(lstm_cell, [input_x], dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

print("Creating network")
embedded = tf.nn.embedding_lookup(embedding, x)
print(embedded)
pred = LSTM(embedded)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()


def to_normalised_words(text):
    return [re.sub(r"[^a-z\-]","",w.lower()) for w in re.split("[\n\s]+",text)]

def get_batch(dataset, start, size):
    xs,ys = [],[]
    end = min(len(dataset.data), start+size)
    for x_raw,y_raw in [(dataset.data[i],dataset.target[i]) for i in range(start, end)]:
        x = [vocab[w] for w in to_normalised_words(x_raw) if w in vocab]
        xs.append(x)
        ys.append(np.array([1 if y_raw == i else 0 for i in range(3)]))
    return np.array(xs),np.array(ys)

print("Ready to train")

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(len(trainset.data)/batch_size)
    offset = 0
    for step in range(epochs):
        batch_x, batch_y = get_batch(trainset,offset,batch_size)
        print(batch_x,batch_x.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
        print("Step " + str(total_batch*i+step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        offset += batch_size
        if offset+batch_size > len(trainset.data):
            offset = random.randint(0,len(trainset.data)-batch_size)

    print("Optimization Finished!")
    test_data,test_label = get_batch(testset,0,len(testset.data))
    test_data = test_data.reshape((len(testset.data), timesteps, embedding_dim))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

