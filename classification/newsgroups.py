from sklearn.datasets import fetch_20newsgroups
import re
from IPython import embed

categories = ["comp.graphics","sci.space","rec.sport.baseball"]
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

def blacklisted(word):
    return "@" in word

PAD_TOKEN = "<pad>"
END_TOKEN = "<end>"

# returns vocab,docs,indexed_docs,doc_lens,sent_lens,labels
def normalise(texts, labels, max_doc_len, max_sent_len, num_classes, debug=False):
    new_vocab = {}
    docs = []
    input_labels = []
    sent_lens = []
    doc_lens = []
    header_re = re.compile(r"^[a-zA-Z_-]+: |^\s*$")
    # start at 1
    ctr = 1
    text_index = 0
    for text in texts:
        doc = []
        doc_sent_lens = []
        
        # remove headers
        lines = text.split("\n")
        while len(lines) > 0 and header_re.match(lines[0]): lines = lines[1:]
        if len(lines) == 0: continue
        new_text = " ".join(lines)
        sentence = []
        # split into sentences and normalise words
        for word in re.split(r"\s+",new_text):
            clean = re.sub(r"[^a-z\-'0-9]","",word.lower())

            # Append the clean word to the sentence
            if not blacklisted(word) and len(clean) > 0:
                sentence.append(clean)

            # Detect if we're at a sentence boundary
            if len(word) > 0 and word[-1] == ".":
                sentence.append("<end>")
                # Add the (padded) sentence to the document if it fits the length requirements
                if len(sentence) <= max_sent_len:
                    doc.append(sentence + [PAD_TOKEN]*(max_sent_len - len(sentence)))
                    assert len(doc[-1]) == max_sent_len
                    doc_sent_lens.append(len(sentence))
                sentence = []
                
            # Add the word to the vocabulary
            if not clean in new_vocab and len(clean) > 0:
                new_vocab[clean] = ctr
                ctr += 1

        # Add the (padded) document to the corpus if it fits the length requirements
        if len(doc) <= max_doc_len and len(doc) > 0:
            docs.append(doc + [[PAD_TOKEN for i in range(max_sent_len)] for j in range(max_doc_len-len(doc))])
            assert(len(docs[-1]) == max_doc_len)
            input_labels.append([1 if i == labels[text_index] else 0 for i in range(num_classes)])
            doc_lens.append(len(doc))
            sent_lens.append(doc_sent_lens+[0]*(max_doc_len-len(doc_sent_lens)))
            if debug:
                print("="*50)
                print("="*50)
                print(text)
                print("="*50)
                print(doc)
                print("="*50)
                print(doc_sent_lens[-1])
                print(doc_lens[-1])
        text_index += 1

    # add special tokens
    new_vocab[END_TOKEN] = ctr
    new_vocab[PAD_TOKEN] = ctr+1

    indexed_docs = [[[new_vocab[word] for word in sent] for sent in doc] for doc in docs]
    
    return new_vocab,docs,indexed_docs,input_labels,doc_lens,sent_lens

#embed()

vocab,text_docs,docs,labels,doc_lens,sent_lens = normalise(newsgroups_train.data,newsgroups_train.target, 20, 25, 3)
rev_vocab = {vocab[word]:word for word in vocab}
