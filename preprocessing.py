import pickle
import spacy
import string

from collections import Counter, defaultdict
from pathlib import Path


def tokenise(data):
    tokeniser = spacy.load('en')
    tokenised_data = []
    for example in data:
        example[2] = example[2].lower()
        example[3] = tokenise_sentence(tokeniser, example[3])
        tokenised_data.append(example)

    return tokenised_data


def tokenise_sentence(tokeniser, sent):
    tokenised = [t.text.lower() for t in tokeniser(sent)]
    
    # REALISED THAT WORD EMBEDDINGS HAVE THESE CONTRACTION ENDINGS SO NO NEED TO DO THIS FOR NOW, JUST USE EMBEDDING
    # deleted_inds = []
    # for i, token in enumerate(tokenised):
        
        # if token.startswith("'"):
        #     tokenised[i-1] += token
        #     deleted_inds.append(i)
        # if token == "n't":
        #     tokenised[i] = "not"

    # num_deleted = 0
    # for j in deleted_inds:
    #     del tokenised[j - num_deleted]
    #     num_deleted += 1

    return [t for t in tokenised if t not in string.punctuation]


def create_vocab(sentences):
    freq = Counter(p for o in sentences for p in o)
    vocab = [o for o, c in freq.most_common(5000)]
    vocab.insert(0, '_unk_')
    string2idx = defaultdict(lambda:0, {v: k for k, v in enumerate(vocab)})

    return vocab, string2idx
    

if __name__ == "__main__":
    t = tokenise_sentence(spacy.load('en'), "Errr, yeah, I haven't been to the loo for quite a time, quite a while, maybe 3 or 4 days")
#     t = tokenise_sentence(spacy.load('en'), "Umm, it's both my hands")
    # print(t)

    # data = pickle.load(Path('./data/data.pkl').open('rb'))
    # tokenised_data = tokenise(data)
    # pickle.dump(tokenised_data, Path('./data/data_tokenised.pkl').open('wb'))

    # tokenised_data = pickle.load(Path('./data/data_tokenised.pkl').open('rb'))
    # sentences = [r[3] for r in tokenised_data]
    # vocab, string2idx = create_vocab(sentences)

    
