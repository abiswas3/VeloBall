import tensorflow as tf
import numpy as np
from IPython import embed
import re, sys
sys.path.append("../")
import newsgroups

trainset = newsgroups.newsgroups_train
testset = newsgroups.newsgroups_test

vocab = newsgroups.vocab
vocab_size = len(vocab.keys());
print(vocab_size)
learning_rate = 0.01
training_epochs = 30
batch_size = 100

l1_size = 10
l2_size = 5
classes = 3

weights = {
    'h1': tf.Variable(tf.random_normal([vocab_size, l1_size])),
    'h2': tf.Variable(tf.random_normal([l1_size, l2_size])),
    'out': tf.Variable(tf.random_normal([l2_size, classes]))
}
bias = {
    'b1': tf.Variable(tf.random_normal([l1_size])),
    'b2': tf.Variable(tf.random_normal([l2_size])),
    'out': tf.Variable(tf.random_normal([classes]))
}
    
def MLP(input_tensor, weights, bias):
    l1_mul = tf.matmul(input_tensor, weights['h1'])
    l1_add = tf.add(l1_mul, bias['b1'])
    l1_act = tf.nn.relu(l1_add)
    
    l2_mul = tf.matmul(l1_act, weights['h2'])
    l2_add = tf.add(l2_mul, bias['b2'])
    l2_act = tf.nn.relu(l2_add)

    l3_mul = tf.matmul(l2_act, weights['out'])
    l3_add = l3_mul + bias['out']

    return l3_add

input_tensor = tf.placeholder(tf.float32,[None, vocab_size],name="input")
output_tensor = tf.placeholder(tf.float32,[None, classes],name="output") 
prediction = MLP(input_tensor, weights, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=output_tensor))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()

def get_batch(dataset, idx, batch_size):
    xs,ys = [],[]
    for x_raw,y_raw in [(dataset.data[i],dataset.target[i]) for i in range(idx*batch_size, (idx+1)*batch_size)]:
        x = np.zeros(vocab_size,dtype=float)
        for w in re.split("[\n\s]+",x_raw):
            w = re.sub(r"[^a-z\-]","",w.lower())
            if w in vocab:
                x[vocab[w]] += 1
        xs.append(x);
        ys.append(np.array([1 if y_raw == i else 0 for i in range(3)]))
    return np.array(xs),np.array(ys)

#embed()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(trainset.data)/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = get_batch(trainset,i,batch_size)
            c,_ = sess.run([loss,optimizer], feed_dict={input_tensor:batch_x,output_tensor:batch_y})
            avg_cost += c / total_batch
        print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_cost))

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(output_tensor, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({input_tensor: batch_x, output_tensor: batch_y}))
        
    total_test_data = len(testset.target)
    batch_x_test,batch_y_test = get_batch(testset,0,total_test_data)
    print("Test Accuracy:", accuracy.eval({input_tensor: batch_x_test, output_tensor: batch_y_test}))
