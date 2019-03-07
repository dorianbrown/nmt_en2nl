#!/bin/bash

echo "Downloading glove word embeddings"
mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove
rm glove.6B.zip

echo "Downloading Europarl Dutch-English parallel corpus"
mkdir -p data/corpus
tar -xvf nl-en.tar -C data/corpus
rm nl-en.tar
