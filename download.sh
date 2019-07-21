#!/bin/bash

EXTERNAL='../External/'
DATA='./data/'

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

# Download XLNet-Base config and vocabulary
wget https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-config.json
mv xlnet-base-case-config.json $DATA
wget https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-spiece.model
mv xlnet-base-cased-spiece.model $DATA

# Download XLNet-Base model
wget https://s3.amazonaws.com/models.huggingface.co/bert/xlnet-base-cased-pytorch_model.bin
mv xlnet-base-cased-pytorch_model.bin $EXTERNAL

python -c "import nltk; nltk.download('stopwords')"
