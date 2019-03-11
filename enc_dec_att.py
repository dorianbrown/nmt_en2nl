#!/usr/bin/env python

import numpy as np
from keras.layers import Embedding, Input, LSTM
from keras.initializers import Constant

from preprocessing import load_nlen_pairs


def embedding_matrix_builder(emb_file, vocab):

    embeddings_index = {}
    with open(emb_file) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    num_words = len(vocab) + 1
    embedding_matrix = np.zeros((num_words, 300))
    for word, i in vocab.items():
        if i > 10_000:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def model_builder():

    embedding_matrix = embedding_matrix_builder('data/glove/glove.6B.300d.txt', en['vocab'])

    embedding = Embedding(len(en['vocab']) + 1,
                          300,
                          embeddings_initializer=Constant(embedding_matrix),
                          input_length=50,
                          trainable=False,
                          mask_zero=True)

    inputs = Input(shape=(50,))
    embeddings = embedding(inputs)
    encoder = LSTM(64, return_sequences=True, unroll=True)(embeddings)

    # Checkout
    # https://github.com/wanasit/katakana/blob/master/notebooks/Attention-based%20Sequence-to-Sequence%20in%20Keras.ipynb
    # for details

    return model


if __name__ == "__main__":

    en, nl = load_nlen_pairs(max_vocab=20_000, max_seq_len=50, max_n=100_000)

    print(f"EN input shape: {en['data'].shape}")
    print(f"NL target shape: {nl['data'].shape}")
