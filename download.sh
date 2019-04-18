#!/bin/bash

EXTERNAL='../External/'

mkdir -p $EXTERNAL

# Download ConceptNet
wget https://s3.amazonaws.com/conceptnet/downloads/2017/edges/conceptnet-assertions-5.5.5.csv.gz
mv conceptnet-assertions-5.5.5.csv.gz $EXTERNAL
gunzip $EXTERNAL/conceptnet-assertions-5.5.5.csv.gz
./src/preprocess.py $EXTERNAL/conceptnet-assertions-5.5.5.csv $EXTERNAL/conceptnet.csv

# Download GloVe vectors
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
mv glove.840B.300d.zip $EXTERNAL
unzip $EXTERNAL/glove.840B.300d.zip

wget http://nlp.stanford.edu/data/glove.6B.zip
mv glove.6B.zip $EXTERNAL
unzip $EXTERNAL/glove.6B.zip

# Download BERT-Base model
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
mv bert-base-uncased.tar.gz $EXTERNAL

python -c "import nltk; nltk.download('stopwords')"
