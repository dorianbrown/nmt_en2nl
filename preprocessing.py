#!/usr/bin/env python

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def load_sentences(file_dir, max_vocab, max_seq_len, max_n):
    with open(file_dir, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]

    lines = lines[:min(max_n, len(lines))]

    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    vocab = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_seq_len)

    return vocab, lines, data


def load_nlen_pairs(max_vocab, max_seq_len, max_n):
    en_vocab, en_text, en_data = load_sentences('data/corpus/europarl-v7.nl-en.en', max_vocab, max_seq_len, max_n)
    nl_vocab, nl_text, nl_data = load_sentences('data/corpus/europarl-v7.nl-en.nl', max_vocab, max_seq_len, max_n)

    en = {
        'vocab': en_vocab,
        'text': en_text,
        'data': en_data
    }
    nl = {
        'vocab': nl_vocab,
        'text': nl_text,
        'data': nl_data
    }

    return en, nl
