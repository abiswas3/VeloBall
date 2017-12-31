import tensorflow as tf
import numpy as np
import sys
sys.path.append("../")
import newsgroups

trainset = newsgroups.newsgroups_train
testset = newsgroups.newsgroups_test
vocab = newsgroups.vocab
vocab_size = len(vocab.keys());
embedding_size = 128
num_classes = 3
num_filters_per_size = 128
filters = {'3':3,'4':4,'5':5}
num_filters_total = num_filters_per_size * len(filters.keys())

weights = {
    'embedding': tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0)),
    'out':tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1))
}
bias = {
    'embedding': tf.Variable(tf.random_normal([embedding_size])),
    'out':tf.Variable(tf.constant(0.1, shape=[num_classes]))
}

    
for f in filters:
    weights[f] = tf.Variable(tf.truncated_normal([filters[f], embedding_size, 1, num_filters_per_size], stddev=0.1))
    bias[f] = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]))

    
def CNN(input_tensor, weights, bias):
    embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
    embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
    pool = {}
    conv = {}
    for f in filters:
        conv[f] = tf.nn.conv2d(embedded_chars_expanded,weights[f],strides=[1, 1, 1, 1],padding="VALID")
        pool[f] = tf.nn.max_pool(tf.nn.relu(tf.nn.bias_add(conv[f], bias[f])),
                                 ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                 strides=[1, 1, 1, 1],padding='VALID')

    h_pool = tf.concat(3, pool.values())
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)
    return tf.nn.xw_plus_b(h_drop, W, b),tf.argmax(scores, 1)


input_data = tf.placeholder(tf.int32, [None, max([len(s) for s in trainset.data])])
input_classes = tf.placeholder(tf.float32, [None, trainset.target[0]])
scores, predictions = CNN(input_data, weights, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(scores, input_classes))
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.argmax(input_classes, 1)), "float"))

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(newsgroups_train.data)/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = get_batch(newsgroups_train,i,batch_size)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor: batch_x,output_tensor:batch_y})
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_cost))

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(output_tensor, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    total_test_data = len(newsgroups_test.target)
    batch_x_test,batch_y_test = get_batch(newsgroups_test,0,total_test_data)
    print("Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))

