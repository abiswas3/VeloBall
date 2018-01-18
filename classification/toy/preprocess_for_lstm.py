import numpy as np
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

def parse_document_into_sentences(x,
                                  min_words_in_sent=5,
                                  max_words_in_sent=10,
                                  min_sents_in_doc=1,
                                  max_sents_in_doc=20):

    '''
    x : document
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
            
        return np.array(document[:max_words_in_sent])
    else:
        return None

if __name__ == '__main__':

    X = np.load('wiki_data/movies/all_movies.npy')
    x = X[5]
    d = parse_document_into_sentences(x)
    # print(d)
    
