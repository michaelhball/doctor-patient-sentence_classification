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
        self.reset()
        while self.i < self.num_examples - 1:
            example = self.data[self.i]
            self.i += 1
            yield example
    
    def _embed_data(self):
        assert(self.we)
        for example in self.data:
            example[3] = self._embed_sentence(example[3])
    
    def _embed_sentence(self, sentence):
        output = []
        for token in sentence:
            if token not in self.we:
                token = 'unk'
            output.append(self.we[token])
        
        return np.array(output)

    def reset(self):
        self.data = tokenise(self.data_reader.read())
        self.num_examples = len(self.data)
        # self.vocab, self.string2idx = create_vocab([['doctor', 'patient']]+[r[3] for r in self.data]) # may not need this if using pretrained embeddings
        if self.we:
            self.data = self.embed_data()
        if self.randomise:
            shuffle(self.data)
        self.i = 0


# if __name__ == "__main__":
#     from data_reader import DataReader
#     dr = DataReader('./data/task_b_interactions_train.tsv')
#     di = DataIterator(dr)

