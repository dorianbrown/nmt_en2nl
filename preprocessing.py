#!/usr/bin/env python

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def load_sentences(file_dir, max_vocab = 20_000, max_seq_len=100, max_n=1e9):
    with open(file_dir, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
        
    lines = lines[:min(len(lines), max_n)]
    
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    
    vocab = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=max_seq_len)
    
    return vocab, lines, data

def load_nlen_pairs():
    en_vocab, en_text, en_data = load_sentences('data/europarl-v7.nl-en.en')
    nl_vocab, nl_text, nl_data = load_sentences('data/europarl-v7.nl-en.nl')
    
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