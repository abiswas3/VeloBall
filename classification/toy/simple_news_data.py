import numpy as np
from collections import Counter

def get_data():
    max_sen_len = 32
    
    with open('DATA/train_txt.txt') as f:
        train_filenames = [i.strip() for i in f.readlines()]

    with open('DATA/train_label.txt') as f:
        labels = [int(i.strip()) for i in f.readlines()]

    LABELS = np.zeros((len(labels), len(np.unique(labels))))
    for i, lab in enumerate(labels):
        LABELS[i,lab] = 1
        
    sentences = []
    for filename in train_filenames:
        with open('DATA/train_txt/{}'.format(filename)) as f:
            sentences.append(f.read().strip())
            
    # Tokenize
    index = 1
    cnt = Counter()
    for sentence in sentences:
        for word in sentence.split():
            word = word.lower()
            cnt[word] += 1            

    num_words = len(cnt.most_common())
    num_acceptable = 1
        
            
    vocabulary = {}
    reverse_vocabulary = {}
    reverse_vocabulary[0] = 'UNK'
            
    index = 1
    for word in cnt:
        if cnt[word] < num_acceptable:
            vocabulary[word] = 0
        else:
            vocabulary[word] = index
            reverse_vocabulary[index] = word
            index +=1

    data = []
    for sentence in sentences:
        row = []
        for word in sentence.split():
            word = word.lower()
            row.append(vocabulary[word])
            
        if len(row) < max_sen_len:
            row = row + [0 for i in range(max_sen_len - len(row))]                
        data.append(row[: max_sen_len])

    return np.array(data[:-1000]), LABELS[:-1000], vocabulary, reverse_vocabulary, np.array(data[-1000:]), LABELS[-1000:]

        
            
            
