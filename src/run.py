#!/usr/bin/env python3
"""
Script to run a baseline model for the SemEval-2018 Task 11 problem (also
COIN 2019 Task 1).
The dataset has a Passage, a Question and a Candidate answer. The target is to
predict if the answer is correct or not.
The baseline uses GloVe to build embeddings for each of the three texts, which
are then encoded using (different) LSTMs. The encoded vectors are then
concatenated and fed into a feed-forward layer that output class probabilities.

This script builds the model, trains it, generates predictions and saves it.
Then it checks if the saving went correctly.
"""
from typing import Dict, Callable, Tuple
import sys
from pathlib import Path
import pickle

import torch
from torch.optim import Adamax
import numpy as np
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder

from models import (BaselineClassifier, AttentiveClassifier, AttentiveReader,
                    SimpleBertClassifier, AdvancedBertClassifier, SimpleTrian,
                    HierarchicalBert, AdvancedAttentionBertClassifier,
                    HierarchicalAttentionNetwork, RelationalTransformerModel,
                    RelationalHan, Dcmn)
from predictor import McScriptPredictor
from util import (example_input, is_cuda, train_model, get_experiment_name,
                  load_data, get_preprocessed_name, parse_cuda, DotDict)
from layers import (lstm_encoder, gru_encoder, lstm_seq2seq, gru_seq2seq,
                    glove_embeddings, learned_embeddings, bert_embeddings,
                    transformer_seq2seq,
                    # cnn_encoder,
                    RelationalTransformerEncoder)
from reader import (SimpleBertReader, SimpleMcScriptReader, SimpleTrianReader,
                    FullTrianReader, McScriptReader, RelationBertReader)


def get_args() -> DotDict:
    usage = """
USAGE:
    run.py CONFIG MODEL [EMBEDDING_TYPE] [CUDA_DEVICE] [NAME] [ENCODER] [TTYPE]

ARGS:
    CONFIG: configuration to use. One of: small, large
    MODEL: model to run. One of: baseline, attentive, reader
    EMBEDDING_TYPE: word embeddings for the text. One of: glove, bert.
    CUDA_DEVICE: device to run the training. -1 for CPU, >=0 for GPU.
    NAME: name for model being trained (used in saving)
    ENCODER: which encoder to use (lstm, gru, transformer)
    TYYPE: transformer type (allen or custom)
"""
    if any('help' in arg or '-h' in arg for arg in sys.argv):
        print(usage)
        exit(0)

    f = DotDict()

    DEFAULT_CONFIG = 'small'  # Can be: _large_ or _small_
    CONFIG = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONFIG

    # Which model to use: 'baseline', 'reader', 'simple-bert', 'advanced-bert',
    #  or 'attentive'.
    DEFAULT_MODEL = 'attentive'
    f.MODEL = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL

    f.NER_EMBEDDING_DIM = 8
    f.REL_EMBEDDING_DIM = 10
    f.POS_EMBEDDING_DIM = 12
    f.HANDCRAFTED_DIM = 7
    DEFAULT_EMBEDDING_TYPE = 'glove'  # can also be 'bert'
    f.EMBEDDING_TYPE = sys.argv[3] if len(
        sys.argv) >= 4 else DEFAULT_EMBEDDING_TYPE

    f.CUDA_DEVICE = parse_cuda(sys.argv[4]) if len(sys.argv) >= 5 else 0
    f.MODEL_NAME = sys.argv[5] if len(sys.argv) >= 6 else None

    DATA_FOLDER = Path('data')
    # Proper configuration path for the External folder. The data one is
    # going to be part of the repo, so this is fine for now, but External isn't
    # always going to be.
    EXTERNAL_FOLDER = Path('..', 'External')

    if CONFIG == 'large':
        # Path to our dataset
        f.TRAIN_DATA_PATH = DATA_FOLDER / 'mctrain-data.json'
        f.VAL_DATA_PATH = DATA_FOLDER / 'mcdev-data.json'
        f.TEST_DATA_PATH = DATA_FOLDER / 'mctest-data.json'
        # Path to our embeddings
        f.GLOVE_PATH = EXTERNAL_FOLDER / 'glove.840B.300d.txt'
        # Size of our embeddings
        f.GLOVE_EMBEDDING_DIM = 300
        # Size of our hidden layers (for each encoder)
        f.HIDDEN_DIM = 50
        f.TRANSFORMER_DIM = 512
        # Size of minibatch
        f.BATCH_SIZE = 24
        # Number of epochs to train model
        f.NUM_EPOCHS = 30
    elif CONFIG == 'small':
        # Path to our dataset
        f.TRAIN_DATA_PATH = DATA_FOLDER / 'small-train.json'
        f.VAL_DATA_PATH = DATA_FOLDER / 'small-dev.json'
        f.TEST_DATA_PATH = DATA_FOLDER / 'small-test.json'
        # Path to our embeddings
        f.GLOVE_PATH = EXTERNAL_FOLDER / 'glove.6B.50d.txt'
        # Size of our embeddings
        f.GLOVE_EMBEDDING_DIM = 50
        # Size of our hidden layers (for each encoder)
        f.HIDDEN_DIM = 50
        f.TRANSFORMER_DIM = 128
        # Size of minibatch
        f.BATCH_SIZE = 2
        # Number of epochs to train model
        f.NUM_EPOCHS = 5

    f.BERT_PATH = EXTERNAL_FOLDER / 'bert-base-uncased.tar.gz'
    f.CONCEPTNET_PATH = EXTERNAL_FOLDER / 'conceptnet.csv'

    # Path to save the Model and Vocabulary
    f.SAVE_FOLDER = Path('experiments')
    f.SAVE_PATH = f.SAVE_FOLDER / \
        get_experiment_name(f.MODEL, CONFIG, f.EMBEDDING_TYPE, f.MODEL_NAME)
    print('Save path', f.SAVE_PATH)

    def preprocessed_name(split_type: str) -> str:
        "Gets the pre-processed pickle filename from the configuration."
        return get_preprocessed_name(split_type, f.MODEL, CONFIG,
                                     f.EMBEDDING_TYPE)

    # Path to save pre-processed input
    f.TRAIN_PREPROCESSED_NAME = preprocessed_name('train')
    f.VAL_PREPROCESSED_NAME = preprocessed_name('val')
    f.TEST_PREPROCESSED_NAME = preprocessed_name('test')

    f.TRAIN_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.TRAIN_PREPROCESSED_NAME
    f.VAL_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.VAL_PREPROCESSED_NAME
    f.TEST_PREPROCESSED_PATH = EXTERNAL_FOLDER / f.TEST_PREPROCESSED_NAME
    print('Pre-processed data path:', f.TRAIN_PREPROCESSED_PATH)

    # Random seed (for reproducibility)
    f.RANDOM_SEED = 1234

    # Model Configuration
    # Use LSTM, GRU or Transformer
    f.ENCODER_TYPE = sys.argv[6] if len(sys.argv) >= 7 else 'lstm'
    f.WHICH_TRANSFORMER = sys.argv[7] if len(sys.argv) >= 8 else 'allen'
    f.BIDIRECTIONAL = True
    f.RNN_LAYERS = 1
    f.RNN_DROPOUT = 0.5 if f.ENCODER_TYPE != 'transformer' else 0
    f.EMBEDDDING_DROPOUT = 0.5 if f.EMBEDDING_TYPE != 'bert' else 0

    # What encoder to use to join the relation embeddings into a single vector.
    f.RELATION_ENCODER = 'cnn'

    return f


args = get_args()


def get_word_embeddings(vocabulary: Vocabulary) -> TextFieldEmbedder:
    if args.EMBEDDING_TYPE == 'glove':
        return glove_embeddings(vocabulary, args.GLOVE_PATH,
                                args.GLOVE_EMBEDDING_DIM, training=True)
    elif args.EMBEDDING_TYPE == 'bert':
        return bert_embeddings(pretrained_model=args.BERT_PATH)
    else:
        raise ValueError(
            f'Invalid word embedding type: {args.EMBEDDING_TYPE}')


def build_dcmn(vocabulary: Vocabulary) -> Model:
    """
    Builds the DCMN.

    Parameters
    ---------
    vocabulary : Vocabulary built from the problem dataset.

    Returns
    -------
    A `DCMN` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocabulary)

    model = Dcmn(
        word_embeddings=word_embeddings,
        vocab=vocabulary,
        embedding_dropout=args.RNN_DROPOUT
    )

    return model


def build_rel_han(vocab: Vocabulary) -> Model:
    """
    Builds the RelationHan.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationHan` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=args.HIDDEN_DIM,
                                      num_layers=args.RNN_LAYERS,
                                      bidirectional=args.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=args.HIDDEN_DIM,
            num_layers=1,
            bidirectional=args.BIDIRECTIONAL,
            dropout=dropout)
        relation_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=args.HIDDEN_DIM,
            num_layers=1,
            bidirectional=args.BIDIRECTIONAL,
            dropout=dropout)
    elif args.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=args.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=args.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )
        relation_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=args.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )

    document_relation_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_encoder.get_output_dim(),
        model_dim=args.TRANSFORMER_DIM,
        feedforward_hidden_dim=args.TRANSFORMER_DIM,
        num_layers=3,
        num_attention_heads=8,
        dropout_prob=0.1,
        return_kb=False
    )

    # Instantiate model with our embedding, encoder and vocabulary
    model = RelationalHan(
        relation_encoder=relation_encoder,
        document_relation_encoder=document_relation_encoder,
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def build_relational_transformer(vocab: Vocabulary) -> Model:
    """
    Builds the RelationalTransformerModel.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationalTransformerModel` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    rel_embeddings = word_embeddings

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=args.HIDDEN_DIM,
                                      num_layers=args.RNN_LAYERS,
                                      bidirectional=args.BIDIRECTIONAL,
                                      dropout=dropout)
        relation_sentence_encoder = encoder_fn(
            input_dim=embedding_dim,
            output_dim=args.HIDDEN_DIM,
            num_layers=args.RNN_LAYERS,
            bidirectional=args.BIDIRECTIONAL,
            dropout=dropout)
    elif args.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=args.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )
        relation_sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=args.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )

    relational_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_sentence_encoder.get_output_dim(),
        model_dim=args.TRANSFORMER_DIM,
        feedforward_hidden_dim=args.TRANSFORMER_DIM,
        num_layers=3,
        num_attention_heads=8,
        dropout_prob=0.1,
        return_kb=False
    )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = RelationalTransformerModel(
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        # document_encoder=document_encoder,
        # relation_encoder=relation_encoder,
        relational_encoder=relational_encoder,
        relation_sentence_encoder=relation_sentence_encoder,
        rel_embeddings=rel_embeddings,
        vocab=vocab,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def build_hierarchical_attn_net(vocab: Vocabulary) -> Model:
    """
    Builds the HierarchicalAttentionNetwork.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `HierarchicalAttentionNetwork` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=args.HIDDEN_DIM,
                                      num_layers=args.RNN_LAYERS,
                                      bidirectional=args.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=args.HIDDEN_DIM,
            num_layers=1,
            bidirectional=args.BIDIRECTIONAL,
            dropout=dropout)
    elif args.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=args.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=args.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=args.TRANSFORMER_DIM,
            ttype=args.WHICH_TRANSFORMER
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = HierarchicalAttentionNetwork(
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def build_advanced_attn_bert(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedBertClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedBertClassifier` model ready to be trained.
    """
    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=args.HIDDEN_DIM,
                             num_layers=args.RNN_LAYERS,
                             bidirectional=args.BIDIRECTIONAL,
                             dropout=dropout)
    elif args.ENCODER_TYPE == 'transformer':
        encoder = transformer_seq2seq(
            input_dim=hidden_dim,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AdvancedAttentionBertClassifier(
        bert_path=args.BERT_PATH,
        encoder=encoder,
        vocab=vocab,
        encoder_dropout=0,
        hidden_dim=hidden_dim
    )

    return model


def build_hierarchical_bert(vocab: Vocabulary) -> Model:
    """
    Builds the HierarchicalBert.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `HierarchicalBert` model ready to be trained.
    """
    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    bert = bert_embeddings(args.BERT_PATH)
    embedding_dim = bert.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=args.HIDDEN_DIM,
                                      num_layers=args.RNN_LAYERS,
                                      bidirectional=args.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=args.HIDDEN_DIM,
            num_layers=1,
            bidirectional=args.BIDIRECTIONAL,
            dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = HierarchicalBert(
        bert_path=args.BERT_PATH,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def build_simple_trian(vocab: Vocabulary) -> Model:
    """
    Builds the TriAN classifier without the extra features (NER, POS, HC).

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AttentiveClassifier` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    rel_embeddings = learned_embeddings(vocab, args.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_q_rel + 2*p_a_rel
    p_input_size = (2*embedding_dim + + 3*args.REL_EMBEDDING_DIM)
    # q_emb
    q_input_size = embedding_dim
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=args.RNN_LAYERS,
                               bidirectional=args.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=args.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=args.BIDIRECTIONAL)
    elif args.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = SimpleTrian(
        word_embeddings=word_embeddings,
        rel_embeddings=rel_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=0,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def build_advanced_bert(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedBertClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedBertClassifier` model ready to be trained.
    """
    rel_embeddings = learned_embeddings(vocab, args.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=args.HIDDEN_DIM,
                             num_layers=args.RNN_LAYERS,
                             bidirectional=args.BIDIRECTIONAL,
                             dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AdvancedBertClassifier(
        bert_path=args.BERT_PATH,
        encoder=encoder,
        rel_embeddings=rel_embeddings,
        vocab=vocab,
        encoder_dropout=0,
        hidden_dim=hidden_dim
    )

    return model


def build_simple_bert(vocab: Vocabulary) -> Model:
    """
    Builds the simple BERT-Based classifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `SimpleBertClassifier` model ready to be trained.
    """
    model = SimpleBertClassifier(bert_path=args.BERT_PATH, vocab=vocab)
    return model


def build_baseline(vocab: Vocabulary) -> Model:
    """
    Builds the Baseline classifier using Glove embeddings and RNN encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `BaselineClassifier` model ready to be trained.
    """
    embeddings = get_word_embeddings(vocab)

    if args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        encoder_fn = lstm_encoder

    embedding_dim = embeddings.get_output_dim()
    encoder = encoder_fn(embedding_dim, args.HIDDEN_DIM,
                         num_layers=args.RNN_LAYERS,
                         bidirectional=args.BIDIRECTIONAL)

    model = BaselineClassifier(embeddings, encoder, vocab)
    return model


def build_attentive_reader(vocab: Vocabulary) -> Model:
    """
    Builds the Attentive Reader using Glove embeddings and GRU encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AttentiveClassifier` model ready to be trained.
    """
    embeddings = get_word_embeddings(vocab)

    embedding_dim = embeddings.get_output_dim()
    p_encoder = gru_seq2seq(embedding_dim, args.HIDDEN_DIM,
                            num_layers=args.RNN_LAYERS,
                            bidirectional=args.BIDIRECTIONAL)
    q_encoder = gru_encoder(embedding_dim, args.HIDDEN_DIM, num_layers=1,
                            bidirectional=args.BIDIRECTIONAL)
    a_encoder = gru_encoder(embedding_dim, args.HIDDEN_DIM, num_layers=1,
                            bidirectional=args.BIDIRECTIONAL)

    model = AttentiveReader(
        embeddings, p_encoder, q_encoder, a_encoder, vocab
    )
    return model


def build_attentive(vocab: Vocabulary) -> Model:
    """
    Builds the Attentive classifier using Glove embeddings and RNN encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AttentiveClassifier` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    pos_embeddings = learned_embeddings(vocab, args.POS_EMBEDDING_DIM,
                                        'pos_tokens')
    ner_embeddings = learned_embeddings(vocab, args.NER_EMBEDDING_DIM,
                                        'ner_tokens')
    rel_embeddings = learned_embeddings(vocab, args.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if args.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif args.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif args.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_pos_emb + p_ner_emb + p_q_rel + 2*p_a_rel
    #       + hc_feat
    p_input_size = (2*embedding_dim + args.POS_EMBEDDING_DIM
                    + args.NER_EMBEDDING_DIM + 3*args.REL_EMBEDDING_DIM
                    + args.HANDCRAFTED_DIM)
    # q_emb + q_pos_emb
    q_input_size = embedding_dim + args.POS_EMBEDDING_DIM
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = args.RNN_DROPOUT if args.RNN_LAYERS > 1 else 0

    if args.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=args.RNN_LAYERS,
                               bidirectional=args.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=args.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=args.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=args.BIDIRECTIONAL)
    elif args.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=args.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AttentiveClassifier(
        word_embeddings=word_embeddings,
        rel_embeddings=rel_embeddings,
        pos_embeddings=pos_embeddings,
        ner_embeddings=ner_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=args.EMBEDDDING_DROPOUT,
        encoder_dropout=args.RNN_DROPOUT
    )

    return model


def test_load(build_model_fn: Callable[[Vocabulary], Model],
              reader: McScriptReader,
              save_path: Path,
              original_prediction: Dict[str, torch.Tensor],
              cuda_device: int) -> None:
    "Test if we can load the model and if its prediction matches the original."
    print('\n>>>>Testing if the model saves and loads correctly')
    # Reload vocabulary
    with open(save_path / 'vocabulary.pickle', 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)
    # Recreate the model.
    model = build_model_fn(vocab)
    # Load the state from the file
    with open(save_path / 'model.th', 'rb') as model_file:
        model.load_state_dict(torch.load(model_file))
    model.eval()
    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    # Try predicting again and see if we get the same results (we should).
    predictor = McScriptPredictor(model, dataset_reader=reader)
    passage, question, answer0, _ = example_input(0)

    _, _, answer1, _ = example_input(1)
    prediction = predictor.predict(
        passage_id="",
        question_id="",
        passage=passage,
        question=question,
        answer0=answer0,
        answer1=answer1
    )
    np.testing.assert_array_almost_equal(
        original_prediction['logits'], prediction['logits'])
    print('Success.')


def create_reader(reader_type: str) -> McScriptReader:
    "Returns the appropriate Reder instance from the type and configuration."
    is_bert = args.EMBEDDING_TYPE == 'bert'
    if reader_type == 'simple':
        return SimpleMcScriptReader(is_bert=is_bert)
    if reader_type == 'full-trian':
        return FullTrianReader(is_bert=is_bert,
                               conceptnet_path=args.CONCEPTNET_PATH)
    if reader_type == 'simple-bert':
        return SimpleBertReader()
    if reader_type == 'simple-trian':
        return SimpleTrianReader(is_bert=is_bert,
                                 conceptnet_path=args.CONCEPTNET_PATH)
    if reader_type == 'relation-bert':
        return RelationBertReader(is_bert=is_bert,
                                  conceptnet_path=args.CONCEPTNET_PATH)
    raise ValueError(f'Reader type {reader_type} is invalid')


def get_modelfn_reader() -> Tuple[Callable[[Vocabulary], Model],
                                  McScriptReader]:
    "Gets the build function and reader for the model"
    # Model -> Build function, reader type
    models = {
        'baseline': (build_baseline, 'simple'),
        'attentive': (build_attentive, 'full-trian'),
        'reader': (build_attentive_reader, 'simple'),
        'simple-bert': (build_simple_bert, 'simple-bert'),
        'advanced-bert': (build_advanced_bert, 'simple-bert'),
        'hierarchical-bert': (build_hierarchical_bert, 'simple-bert'),
        'simple-trian': (build_simple_trian, 'simple-trian'),
        'advanced-attn-bert': (build_advanced_attn_bert, 'simple-bert'),
        'han': (build_hierarchical_attn_net, 'simple-bert'),
        'rtm': (build_relational_transformer, 'relation-bert'),
        'relhan': (build_rel_han, 'relation-bert'),
        'dcmn': (build_dcmn, 'simple'),
    }

    if args.MODEL in models:
        build_fn, reader_type = models[args.MODEL]
        return build_fn, create_reader(reader_type)
    raise ValueError(f'Invalid model name: {args.MODEL}')


def make_prediction(model: Model, reader: McScriptReader) -> torch.Tensor:
    "Create a predictor to run our model and get predictions."
    model.eval()
    predictor = McScriptPredictor(model, reader)

    print()
    print('#'*5, 'EXAMPLE', '#'*5)
    passage, question, answer1, label1 = example_input(0)
    _, _, answer2, _ = example_input(1)
    result = predictor.predict("", "", passage, question, answer1, answer2)
    prediction = np.argmax(result['prob'])

    print('Passage:\n', '\t', passage, sep='')
    print('Question:\n', '\t', question, sep='')
    print('Answers:')
    print('\t1:', answer1)
    print('\t2:', answer2)
    print('Prediction:', prediction+1)
    print('Correct:', 1 if label1 == 1 else 2)

    return result


def run_model() -> None:
    "Execute model according to the configuration"
    # Which model to use?
    build_fn, reader = get_modelfn_reader()

    def optimiser(model: Model) -> torch.optim.Optimizer:
        return Adamax(model.parameters(), lr=2e-3)

    # Create SAVE_FOLDER if it doesn't exist
    args.SAVE_FOLDER.mkdir(exist_ok=True, parents=True)
    train_dataset = load_data(data_path=args.TRAIN_DATA_PATH,
                              reader=reader,
                              pre_processed_path=args.TRAIN_PREPROCESSED_PATH)
    val_dataset = load_data(data_path=args.VAL_DATA_PATH,
                            reader=reader,
                            pre_processed_path=args.VAL_PREPROCESSED_PATH)
    test_dataset = load_data(data_path=args.TEST_DATA_PATH,
                             reader=reader,
                             pre_processed_path=args.TEST_PREPROCESSED_PATH)

    model = train_model(build_fn,
                        train_data=train_dataset,
                        val_data=val_dataset,
                        test_data=test_dataset,
                        save_path=args.SAVE_PATH,
                        num_epochs=args.NUM_EPOCHS,
                        batch_size=args.BATCH_SIZE,
                        optimiser_fn=optimiser,
                        cuda_device=args.CUDA_DEVICE)

    result = make_prediction(model, reader)

    print('Save path', args.SAVE_PATH)

    cuda_device = 0 if is_cuda(model) else -1
    test_load(build_fn, reader, args.SAVE_PATH, result, cuda_device)


if __name__ == '__main__':
    torch.manual_seed(args.RANDOM_SEED)

    run_model()
