from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd

'''Returns bag of words vectors with each element weighted by tf-idf
score instead of 1
'''


def train(document_list, stop_words='english'):


    if len(stop_words) > 0:
        count_vectorizer = CountVectorizer(stop_words=stop_words)
    else:
        count_vectorizer = CountVectorizer()
        
    count_vectorizer.fit_transform(document_list)
    
    return count_vectorizer

def get_features(count_vectorizer, document_list):

    # Build bag of words model for documents
    # using words in count_vectorizer
    freq_term_matrix = count_vectorizer.transform(document_list)

    
    # rewweight the words based on occurrence across documents
    # tf(t,d): times term t ocurs in document t
    # idf(t) : times term occures in all documents
    # tf-idf(t,d) : tf(t,d)*idf(t)
    
    tfidf = TfidfTransformer(norm="l2")
    tfidf.fit(freq_term_matrix)

    tf_idf_matrix = tfidf.transform(freq_term_matrix)
    return np.array(tf_idf_matrix.todense())

if __name__ == '__main__':

    # Unit test
    train_set = ["The sky is blue.",
                 "The sun is bright MESSI."]

    test_set = ["The sun in the sky is bright MESSI",
                "We can see the shining sun, the bright sun."
]
    # Build vocabulary counts
    c = train(train_set, stop_words="")

    # Use counts in vocaubulary to get tfidfs scores for each word in
    # document
    feat_vecs = get_features(c, test_set)

    # Print nicely
    X = pd.DataFrame(feat_vecs, columns=c.get_feature_names())
    print(X)

