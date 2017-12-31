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

timesteps = 20
num_hidden = 128
num_classes = 3
epochs = 20

trainset = newsgroups.newsgroups_train
testset = newsgroups.newsgroups_test
vocab = newsgroups.vocab
vocab_size = len(vocab)
embedding_dim = 128
embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0))
maxlen = max([len(doc) for doc in trainset.data])

x = tf.placeholder(tf.int64, shape=[batch_size, maxlen])
y = tf.placeholder(tf.int32, [batch_size, num_classes])
s = tf.placeholder(tf.int32, [batch_size])
dropout_keep_prob = tf.placeholder(tf.float32)

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def LSTM(input_x, seqlens):
    print("I",input_x)
    print("S",seqlens)
    lstm_cell = rnn.BasicLSTMCell(num_hidden)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, input_x, sequence_length=seqlens, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

print("Creating network")
embedded = tf.nn.embedding_lookup(embedding, x)
embed()
pred = LSTM(embedded, s)
print(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()


def to_normalised_words(text):
    return [re.sub(r"[^a-z\-]","",w.lower()) for w in re.split("[\n\s]+",text)]

#returns X,Y,seqlens
def get_batch(dataset, start, size):
    xs,ys,ss = [],[],[]
    end = min(len(dataset.data), start+size)
    for x_raw,y_raw in [(dataset.data[i],dataset.target[i]) for i in range(start, end)]:
        words = to_normalised_words(x_raw)
        x = [vocab[w] for w in words if w in vocab]+[0]*(maxlen-len(words))
        xs.append(np.array(x))
        ys.append(np.array([1 if y_raw == i else 0 for i in range(3)]))
        ss.append(len(x))
    return np.array(xs),np.array(ys),np.array(ss)

print("Ready to train")

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(len(trainset.data)/batch_size)
    offset = 0
    for step in range(epochs):
        batch_x, batch_y, batch_s = get_batch(trainset,offset,batch_size)
        # batch_x = ((doc0_word0,doc0_word1,doc0_word2,...),...,(doc255_word0,doc255_word1,doc255_word2,...))
        # batch_y = ((doc0_is_class0,,doc0_is_class1,doc0_is_class2),...,(doc255_is_class0,,doc255_is_class1,doc255_is_class2))
        # batch_s = (doc0_len,...,doc255_len)
        print("X",batch_x[-1],batch_x[0].shape,batch_x.shape)
        print("Y",batch_y[-1],batch_y[0].shape,batch_y.shape)
        print("S",batch_s[-1],batch_s[0].shape,batch_s.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, s: batch_s})
        acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y, s: batch_s})
        print("Step " + str(total_batch*i+step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        offset += batch_size
        if offset+batch_size > len(trainset.data):
            offset = random.randint(0,len(trainset.data)-batch_size)

    print("Optimization Finished!")
    test_data,test_label = get_batch(testset,0,len(testset.data))
    test_data = test_data.reshape((len(testset.data), timesteps, embedding_dim))
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))

