import gensim.downloader as api
import numpy as np

from random import shuffle

from preprocessing import create_vocab, tokenise


class DataIterator():
    def __init__(self, data_reader, word_embedding_source=None, randomise=True):
        self.data_reader = data_reader
        self.we = api.load(word_embedding_source) if word_embedding_source else None
        self.randomise = randomise
        self.reset()

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        if self.randomise:
            shuffle(self.data)
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example
    
    def _embed_data(self):
        assert(self.we)
        for i, example in enumerate(self.data):
            self.data[i][3] = self._embed_sentence(example[3])
    
    def _embed_sentence(self, sentence):
        output = []
        for token in sentence:
            if token not in self.we:
                token = 'unk'
            output.append(self.we[token])
        
        return np.array(output)

    def _labels_to_idx(self):
        self.simple_labels = []
        self.extended_labels = []
        for i, example in enumerate(self.data):
            if example[4] not in self.simple_labels:
                self.simple_labels.append(example[4])
            self.data[i][4] = self.simple_labels.index(example[4])
            if example[5] not in self.extended_labels:
                self.extended_labels.append(example[5])
            self.data[i][5] = self.extended_labels.index(example[5])

    def reset(self):
        self.data = tokenise(self.data_reader.read())
        self.num_examples = len(self.data)
        self._labels_to_idx()
        # self.vocab, self.string2idx = create_vocab([['doctor', 'patient']]+[r[3] for r in self.data]) # may not need this if using pretrained embeddings
        if self.we:
            self._embed_data()
        if self.randomise:
            shuffle(self.data)
        self.i = 0
