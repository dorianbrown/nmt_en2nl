#!/bin/bash

echo "Downloading glove word embeddings"
mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove
rm glove.6B.zip

echo "Downloading Europarl Dutch-English parallel corpus"
mkdir -p data/corpus
wget http://www.statmt.org/europarl/v7/nl-en.tgz
tar -xvzf nl-en.tgz -C data/corpus
rm nl-en.tgz
