from .basic_classifier import BasicClassifier
from .encoder import Encoder
from .linear_block import LinearBlock
from .lstm_encoder import LSTMEncoder
from .pooling_classifier import PoolingClassifier
from .pooling_encoder import PoolingEncoder
from .rnn_pooling_classifier import RNNPoolingClassifier

import torch.nn as nn


encoders = {
    "max_pool_embeddings": PoolingEncoder("max"),
    "ave_pool_embeddings": PoolingEncoder("ave"),
    "basic": Encoder(),
    "lstm": LSTMEncoder
}

classifiers = {
    "basic": BasicClassifier,
    "pooling": PoolingClassifier,
    "rnn_pooling": RNNPoolingClassifier,
}


def create_classifier(layers, drops, encoder_type="basic", classifier_model="pooling", args=None):
    encoder = encoders[encoder_type]
    if encoder_type == "lstm":
        encoder = encoder(args['embedding_dim'], args['hidden_dim'])
    classifier = classifiers[classifier_model](layers, drops)

    return nn.Sequential(encoder, classifier)