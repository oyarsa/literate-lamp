#!/bin/bash

EXTERNAL='../External/'

mkdir -p $EXTERNAL

# Download ConceptNet
wget https://s3.amazonaws.com/conceptnet/downloads/2017/edges/conceptnet-assertions-5.5.5.csv.gz
gunzip conceptnet-assertions-5.5.5.csv.gz
mv conceptnet-assertions-5.5.5.csv $EXTERNAL

# Download GloVe vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
mv glove.840B.300d.txt $EXTERNAL

# Download BERT-Base model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
mv bert-base-uncase.tar.gz $EXTERNAL
