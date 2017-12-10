import collections
import numpy as np
import math
import random
import tensorflow as tf

class Word2Vec(object):

    def __init__(self, vocabulary, vocabulary_size):

        self.vocabulary_size = vocabulary_size

        self.data, self.count, self.dictionary, self.reverse_dictionary = self.build_dataset(vocabulary,
                                                                                             vocabulary_size)

        self.batch_size     = 128
        self.embedding_size = 128     # Dimension of the embedding vector.
        self.skip_window    = 1       # How many words to consider left and right.
        self.num_skips      = 2       # How many times to reuse an input to generate a label.
        self.num_sampled    = 64      # Number of negative examples to sample.


        self.graph = tf.Graph()
        
        self.create_network()
        
        self.sess = tf.Session(graph=self.graph)
        
        
    def build_dataset(self, words, n_words):
        """Process raw inputs into a dataset.
        Keeps most common words only 

        Params:
        words: all the words in the vocabulary
        n_words: vocab size limit
        """
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(n_words - 1))
        
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
            
        data = list()
        unk_count = 0
        for word in words:

            # if I can't find word: index of word = 0
            index = dictionary.get(word, 0)
            if index == 0:  # dictionary['UNK']
                unk_count += 1
                
            data.append(index)
            
        count[0][1] = unk_count
        
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        return data, count, dictionary, reversed_dictionary

    # Step 3:
    def generate_batch(self,
                       data,
                       count,
                       dictionary,
                       reversed_dictionary,
                       batch_size,
                       num_skips,
                       skip_window,
                       data_index):

        '''
        Function to generate a training batch for the skip-gram model.

        Params:
        data                : list of vocabulary in terms of indexes not words
        count               : count of each word
        dictionary          : maps words to indices
        reversed_dictionary : maps indices to words
        batch_size          : size of training batch
        num_skips           : number of times we reuse a target word to generate context
        skip_window         : size of window left/right side of target (total window size is
        2*skip_window + 1
        '''
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * skip_window

        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        span = 2 * skip_window + 1  # [ skip_window target skip_window ]

        buffer = collections.deque(maxlen=span)
        if data_index + span > len(data):
            data_index = 0

        buffer.extend(data[data_index:data_index + span])
        data_index += span
        for i in range(batch_size // num_skips):
            context_words = [w for w in range(span) if w != skip_window]
            words_to_use = random.sample(context_words, num_skips)

            for j, context_word in enumerate(words_to_use):
                batch[i * num_skips + j] = buffer[skip_window]
                labels[i * num_skips + j, 0] = buffer[context_word]

            if data_index == len(data):
                buffer[:] = data[:span]
                data_index = span
            else:
                buffer.append(data[data_index])
                data_index += 1
                # Backtrack a little bit to avoid skipping words in the end of a batch

        data_index = (data_index + len(data) - span) % len(data)
        return batch, labels, data_index


    def create_network(self):
        
        with self.graph.as_default():
            
            self.train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Ops and variables pinned to the CPU because of missing GPU implementation
            with tf.device('/cpu:0'):
                
                # Initialize words with uniform (-1,1)
                embeddings = tf.Variable(
                    tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
                )

                # Look up embeddings for inputs.
                embed = tf.nn.embedding_lookup(embeddings, self.train_inputs)

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(
                    tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                        stddev=1.0 / math.sqrt(self.embedding_size)))

                nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

                # Compute the average NCE loss for the batch.
                # tf.nce_loss automatically draws a new sample of the negative labels each
                # time we evaluate the loss.
                # Explanation of the meaning of NCE loss:
                #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
                self.loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=self.train_labels,
                                   inputs=embed,
                                   num_sampled=self.num_sampled,
                                   num_classes=self.vocabulary_size))


                # Compute the cosine similarity between minibatch examples and all embeddings.
                norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
                self.normalized_embeddings = embeddings / norm

                # Everything in the graph that is variables will get initialized when we run this
                self.init = tf.global_variables_initializer()

                # Construct the SGD optimizer using a learning rate of 1.0.
                self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)



    def train(self):

        data_index = 0
        num_steps = 100001

        self.sess = tf.Session(graph=self.graph)
        self.init.run(session=self.sess)
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels, data_index = self.generate_batch(self.data,
                                                                    self.count,
                                                                    self.dictionary,
                                                                    self.reverse_dictionary,
                                                                    self.batch_size,
                                                                    self.num_skips,
                                                                    self.skip_window,
                                                                    data_index)

            feed_dict = {self.train_inputs: batch_inputs,
                         self.train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()

            _, loss_val = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0


            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                soln = self.normalized_embeddings.eval(session=self.sess)

        self.sess.close()

        self.word_embeddings = soln



