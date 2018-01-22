import numpy as np
import nltk
nltk.download('punkt')
import sys
sys.path.append('../../')

from nltk.tokenize import sent_tokenize, word_tokenize
import json
from collections import Counter
import operator

def parse_document_into_sentences(x,
                                  min_words_in_sent=5,
                                  max_words_in_sent=10,
                                  min_sents_in_doc=1,
                                  max_sents_in_doc=20):

    '''
    x : document in regular english form
    '''
    sentences = sent_tokenize(x)
    document = []
    for s in sentences:
        new_sentence = []
        # might change this with regex
        for w in word_tokenize(s):
            new_sentence.append(w)

        if len(new_sentence) >= min_words_in_sent:

            if len(new_sentence) < max_words_in_sent:
                new_sentence += ['UNK' for i in range(max_words_in_sent  - len(new_sentence))]

            document.append(new_sentence[:max_words_in_sent])



    if len(document) >= min_words_in_sent :
        if len(document) < max_words_in_sent:
            document += [['UNK' for i in range(max_words_in_sent)]
                         for i in range(max_sents_in_doc - len(document))]

        return document[:max_words_in_sent]
    else:
        return None

def naive_corpus(lst_of_all_words):

    x = Counter(lst_of_all_words)
    sorted_x = sorted(x.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

def tokenized(document, d):

    doc = []
    num_sents = len(document)
    for i,s in enumerate(document):
        if s[0] == 'UNK':
            num_sents = i
            break

    sCount = []        
    for i,s in enumerate(document):
        j = 0
        sent = []
        for w in s:

            # if w not in d:
            #     return None, None, None
            sent.append(d[w])            
            if w != 'UNK':
                j +=1

        sCount.append(j)
        doc.append(sent)

    sCount = np.array(sCount)
    return np.array(doc), sCount, num_sents

if __name__ == '__main__':

    pass
    # print('Building global counts')
    # global_count = Counter([j for i in lst for j in i])

    # print('\nGenerating idf values')
    # for ind, word in enumerate(global_count):
    #     print(ind)
    #     temp = global_count[word]
    #     global_count[word] = {'total': temp, 'idf': idf(word, len(lstx), lst)}
