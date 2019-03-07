#!/bin/bash

mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d data/glove
rm glove.6B.zip
