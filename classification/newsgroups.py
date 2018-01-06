from sklearn.datasets import fetch_20newsgroups
import re

categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

def make_index(texts):
    new_vocab = {}

    # start at 1
    ctr = 1
    for text in texts:
        for word in re.split("[\n\s]+",text):
            word = re.sub(r"[^a-z\-]","",word.lower())
            if not word in new_vocab and len(word) > 0:
                new_vocab[word] = ctr
                ctr += 1
                
    return new_vocab

vocab = make_index(newsgroups_train.data + newsgroups_test.data)
vocab_size = len(vocab.keys());
