import gensim.downloader as api
import numpy as np
import pickle

from pathlib import Path
from random import shuffle

from preprocessing import create_vocab, tokenise


class PickleDataIterator():
    def __init__(self, data_file, randomise):
        self.data_file = data_file
        self.randomise = randomise
        self.simple_labels = pickle.load(Path('./data/simple_labels.pkl').open('rb'))
        self.extended_labels = pickle.load(Path('./data/extended_labels.pkl').open('rb'))
        self.fetch_data()
    
    def __len__(self):
        return self.num_examples
    
    def __iter__(self):
        self.reset()
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example

    def fetch_data(self):
        self.data = pickle.load(Path(self.data_file).open('rb'))
        self.num_examples = len(self.data)
        self.reset()

    def reset(self):
        self.i = 0
        if self.randomise:
            shuffle(self.data)


class DataIterator():
    def __init__(self, data_reader, word_embedding_source=None, randomise=True, bow=False):
        self.data_reader = data_reader
        self.we = api.load(word_embedding_source) if word_embedding_source else None
        self.randomise = randomise
        self.bow = bow
        self.fetch_data()

    def __len__(self):
        return self.num_examples

    def __iter__(self):
        self.reset()
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example
    
    def _embed_data(self):
        assert(self.we)
        for i, example in enumerate(self.data):
            self.data[i][2] = self.we[example[2]]
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
    
    def _bow_data(self):
        assert(self.bow)
        for i, x in enumerate(self.data):
            self.data[i][2] = self.string2idx[x[2]]
            self.data[i][3] = np.array([self.string2idx[w] for w in x[3]])

    def fetch_data(self):
        self.data = tokenise(self.data_reader.read())
        self.num_examples = len(self.data)
        self._labels_to_idx()
        self.vocab, self.string2idx = create_vocab([['doctor', 'patient']]+[x[3] for x in self.data])
        if self.bow:
            self._bow_data()
        elif self.we:
            self._embed_data()
        self.reset()

    def reset(self):
        self.i = 0
        if self.randomise:
            shuffle(self.data)
