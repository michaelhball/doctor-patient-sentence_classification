from .basic_classifier import BasicClassifier
from .linear_block import LinearBlock
from .pooling_encoder import PoolingEncoder

import torch.nn as nn


encoders = {
    "max_pool_embeddings": PoolingEncoder("max"),
    "ave_pool_embeddings": PoolingEncoder("ave"),
}

classifiers = {
    "basic": BasicClassifier
}


def create_classifier(layers, drops, encoder_type="max_pool_embeddings", class_type="basic"):
    encoder = encoders[encoder_type]
    classifier = classifiers[class_type](layers, drops)

    return nn.Sequential(encoder, classifier)