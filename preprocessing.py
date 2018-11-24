import spacy

from collections import Counter, defaultdict


def tokenise(data):
    """
    Tokenises input data (each row is one example from dataset).
    """
    tokeniser = spacy.load('en')
    tokenised_data = []
    for example in data:
        example[2] = example[2].lower()
        example[3] = tokenise_sentence(tokeniser, example[3])
        tokenised_data.append(example)

    return tokenised_data


def tokenise_sentence(tokeniser, sent):
    """
    Tokenises sentence using Spacy tokeniser (found to work well even for 
        poorly-formed English sentences)
    """
    return [t.text.lower() for t in tokeniser(sent)]


def create_vocab(sentences):
    """
    Creates a vocab (and the inverse dictionary) given a 
        list of tokenised sentences.
    """
    freq = Counter(p for o in sentences for p in o)
    vocab = [o for o, c in freq.items()]
    vocab.insert(0, '_pad_')
    vocab.insert(0, '_unk_')
    string2idx = defaultdict(lambda:0, {v: k for k, v in enumerate(vocab)})

    return vocab, string2idx
