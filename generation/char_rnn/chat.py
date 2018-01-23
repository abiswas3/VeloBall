import numpy as np
import random

BEGIN = 256
END=257
PAD=258
DIM=259

class chat:
    def __init__(self, filename, maxlen):
        raw_data = []
        self.maxlen = maxlen
        with open(filename,"rb") as f:
            s = f.read()
            while len(s) > 0:
                #print(len(s))
                idx = s.find(ord("\n"))
                line,s = s[:idx],s[idx+1:]
                if len(line) > 0:
                    raw_data.append(line)
        self.data = []
        self.lengths = []
        self.outputs = []
        for x in raw_data:
            l,v = self.get_input(x)
            o = self.get_output(x)
            self.data.append(v)
            self.lengths.append(l)
            self.outputs.append(o)

    def to_string(self, v):
        s = []
        v = np.array(v)
        for x in v:
            c = np.argmax(x)
            if c > 0 and c < 128:
                s.append(chr(c))
            elif c == BEGIN: s.append("<BEG>")
            elif c == END: s.append("<END>")
            elif c == PAD: s.append("<PAD>")
        return "".join(s)
            
    def get_input(self, data):
        #tokenised = [BEGIN]+list(data[:self.maxlen-2])+[END]
        tokenised = list(data[:self.maxlen-2])+[END]
        padded = tokenised + [PAD]*(self.maxlen - len(tokenised))
        return len(tokenised),[[1 if i == x else 0 for i in range(DIM)] for x in padded]

    def get_output(self, data):
        tokenised = list(data[1:self.maxlen-1])+[END]
        padded = tokenised + [PAD]*(self.maxlen - len(tokenised))
        return [[1 if i == x else 0 for i in range(DIM)] for x in padded]

    def get_batch(self, batch_size):
        indices = random.sample(list(range(len(self.data))),batch_size)
        return [self.data[i] for i in indices],[self.lengths[i] for i in indices],[self.outputs[i] for i in indices]
