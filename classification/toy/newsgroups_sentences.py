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

# returns vocab,sentences,indexed_sentences,sent_lens,labels
def normalise(texts, labels, min_sent_len, max_sent_len, num_classes, debug=False):
    new_vocab = {}
    sentences = []
    input_labels = []
    sent_lens = []
    header_re = re.compile(r"^[a-zA-Z_-]+: |^\s*$")
    # start at 1
    vocab_ctr = 0
    doc_ctr = 0
    for text in texts:
        
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
                if min_sent_len <= len(sentence) <= max_sent_len:
                    sentences.append(sentence + [PAD_TOKEN]*(max_sent_len - len(sentence)))
                    assert len(sentences[-1]) == max_sent_len
                    input_labels.append([1 if i == labels[doc_ctr] else 0 for i in range(num_classes)])
                    sent_lens.append(len(sentence))
                sentence = []
                
            # Add the word to the vocabulary
            if not clean in new_vocab and len(clean) > 0:
                new_vocab[clean] = vocab_ctr
                vocab_ctr += 1
        doc_ctr += 1
        if debug:
            print("="*50)
            print("="*50)
            print(text)
            print("="*50)
            print(sentence)
            print("="*50)
            print(sent_lens[-1])
            print([-1])

    # add special tokens
    new_vocab[END_TOKEN] = vocab_ctr
    new_vocab[PAD_TOKEN] = vocab_ctr+1

    indexed_sentences = [[new_vocab[word] for word in sent] for sent in sentences]
    
    return new_vocab,sentences,indexed_sentences,input_labels,sent_lens

vocab,text_sentences,sentences,labels,sent_lens = normalise(newsgroups_train.data,newsgroups_train.target, 10, 32, 3)
rev_vocab = {vocab[word]:word for word in vocab}
