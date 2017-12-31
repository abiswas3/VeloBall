import numpy as np
import random
import re

def generate_sentence(length):

    '''
    Generate a sentence of size length
    A sentence is of the form [ABCDE]+.

    Also output score for that sentence
    '''
    some_big_number = 100000
    candidates  = ['A', 'B', 'C', 'D', 'E']
        
    sentence = ''.join(random.choices(candidates, k=length))+'.'

    importance = len(re.findall(r'E', sentence))

    start_happy = some_big_number
    if re.search(r'AB', sentence):
        start_happy = re.search(r'AB', sentence).span()[0]

    start_sad = some_big_number
    if re.search(r'AC', sentence):
        start_sad = re.search(r'AC', sentence).span()[0]

    label = 0
    if start_happy < start_sad:
        label = +1
    elif start_happy > start_sad:
        label = -1
                    

    return sentence, label, importance

def generate_doc(num_sents, max_sent_len=20, min_sent_len=5):

    doc = []
    document_score = 0
    for i in range(num_sents):
        sen_length = random.randint(min_sent_len, max_sent_len)    
        sen, lab, imp = generate_sentence(sen_length)
        document_score += lab*imp
        doc.append(sen)

    return ''.join(doc), np.sign(document_score)
        

def generate_data(num_docs,
                  max_sent=50,
                  min_sent=10,
                  max_sent_len=20,
                  min_sent_len=5):

    '''
    Args:
    num_docs     : number of documents
    max_sent=50  : maximum number of sentences in document
    min_sent=10, : minimum number of sentences in document
    max_sent_len : maximum words in a sentence
    min_sent_len : minimum words in a sentence

    Output:
    docs         : list of documents
    labels       : list of labels for each document
    
    '''
    docs = []
    labels = []
    num_pos = 0
    num_neg = 0
    for i in range(num_docs):
        num_sents = random.randint(min_sent, max_sent)
        
        d, lab = generate_doc(num_sents,
                              max_sent_len=max_sent_len,
                              min_sent_len=min_sent_len)

        docs.append(d)
        labels.append(lab)
        if lab == 1:
            num_pos +=1
        else:
            num_neg +=1

    print('Num sad: {}\nNum happy: {}'.format(num_neg, num_pos))
    
    return docs, labels

if __name__ == '__main__':

    # s, label, imp = generate_sentence(20)
    # print(s)
    # print(label)
    # print(imp)

    # num_sents = 5
    # d, lab = generate_doc(num_sents, max_sent_len=20, min_sent_len=5)

    num_docs = 3
    data, labels = generate_data(num_docs,
                                 max_sent=50,
                                 min_sent=10,
                                 max_sent_len=20,
                                 min_sent_len=5)
