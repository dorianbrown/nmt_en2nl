#!/bin/bash

mkdir -p data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
tar -xvf glove.6B.zip -C data/glove
rm glove.6B.zip
