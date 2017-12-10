import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import sys, re
sys.path.append("../")
import newsgroups
from IPython import embed

learning_rate = 0.001
batch_size = 128
display_step = 200

timesteps = 100
num_hidden = 128
num_classes = 3

trainset = newsgroups.newsgroups_train
testset = newsgroups.newsgroups_test
vocab = newsgroups.vocab
embedding_dim = 128
embedding = {w:np.random.random_sample(embedding_dim)*20-10 for w in vocab}

x = tf.placeholder("float", [None, timesteps,embedding_dim])
y = tf.placeholder("float", [None, num_classes])

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def LSTM(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

print("Creating network")
pred = LSTM(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()



def get_batch(dataset, idx, batch_size):
    xs,ys = [],[]
    for x_raw,y_raw in [(dataset.data[i],dataset.target[i]) for i in range(idx*batch_size, (idx+1)*batch_size)]:
        x = []
        for w in re.split("[\n\s]+",x_raw):
            w = re.sub(r"[^a-z\-]","",w.lower())
            if w in embedding:
                x.append(embedding[w])
                
        x = (x + [np.zeros(embedding_dim) for i in range(timesteps)])[:timesteps]
        xs.append(x)
        ys.append(np.array([1 if y_raw == i else 0 for i in range(3)]))
    return np.array(xs),np.array(ys)

print("Ready to train")

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(len(trainset.data)/batch_size)
    for i in range(5):
        for step in range(total_batch):
            batch_x, batch_y = get_batch(trainset,step,batch_size)
            #embed()
            batch_x = batch_x.reshape((batch_size, timesteps, embedding_dim))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y})
            print("Step " + str(total_batch*i+step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

    print("Optimization Finished!")
    test_data,test_label = get_batch(testset,0,len(testset.data))
    test_data = test_data.reshape((len(testset.data), timesteps, embedding_dim))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

