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
from typing import Dict, Callable, Tuple, List
from pathlib import Path
from collections import defaultdict
import pickle

import torch
from torch.optim import Adamax
import numpy as np
from allennlp.training.util import evaluate as allen_eval
from allennlp.models import Model
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.instance import Instance
from allennlp.modules.text_field_embedders import TextFieldEmbedder

import args
from models import (BaselineClassifier, Trian, AttentiveReader,
                    SimpleBertClassifier, AdvancedBertClassifier, SimpleTrian,
                    HierarchicalBert, AdvancedAttentionBertClassifier,
                    HierarchicalAttentionNetwork, RelationalTransformerModel,
                    RelationalHan, Dcmn, ZeroTrian, SimpleXLNetClassifier,
                    AdvancedXLNetClassifier, RelationalXL)
from predictor import McScriptPredictor
from util import example_input, is_cuda, train_model, load_data
from layers import (lstm_encoder, gru_encoder, lstm_seq2seq, gru_seq2seq,
                    glove_embeddings, learned_embeddings, bert_embeddings,
                    transformer_seq2seq, xlnet_embeddings,
                    RelationalTransformerEncoder)
from readers import (SimpleBertReader, SimpleMcScriptReader,
                     SimpleTrianReader, FullTrianReader,
                     BaseReader, RelationBertReader, SimpleXLNetReader,
                     RelationXLNetReader)


ARGS = args.get_args()


def get_word_embeddings(vocabulary: Vocabulary) -> TextFieldEmbedder:
    "Instatiates the word embeddings based on config."
    if ARGS.EMBEDDING_TYPE == 'glove':
        return glove_embeddings(vocabulary, ARGS.GLOVE_PATH,
                                ARGS.GLOVE_EMBEDDING_DIM, training=True)
    if ARGS.EMBEDDING_TYPE == 'bert':
        return bert_embeddings(pretrained_model=ARGS.BERT_PATH,
                               training=ARGS.finetune_embeddings)
    if ARGS.EMBEDDING_TYPE == 'xlnet':
        return xlnet_embeddings(config_path=ARGS.xlnet_config_path,
                                model_path=ARGS.xlnet_model_path,
                                training=ARGS.finetune_embeddings)
    raise ValueError(
        f'Invalid word embedding type: {ARGS.EMBEDDING_TYPE}')


def build_relational_xl(vocabulary: Vocabulary) -> Model:
    """
    Builds the RelationalXL.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `RelationalXL` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocabulary)
    embedding_dim = word_embeddings.get_output_dim()

    if ARGS.ENCODER_TYPE == 'lstm':
        seq_fn = lstm_seq2seq
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        seq_fn = gru_seq2seq
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        relation_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        text_encoder = seq_fn(input_dim=embedding_dim,
                              output_dim=ARGS.HIDDEN_DIM,
                              num_layers=ARGS.RNN_LAYERS,
                              bidirectional=ARGS.BIDIRECTIONAL,
                              dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = RelationalXL(
        word_embeddings=word_embeddings,
        text_encoder=text_encoder,
        relation_encoder=relation_encoder,
        vocab=vocabulary,
        encoder_dropout=0.5
    )

    return model


def build_advanced_xlnet(vocab: Vocabulary) -> Model:
    """
    Builds the AdvancedXLNetClassifier.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `AdvancedXLNetClassifier` model ready to be trained.
    """
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    else:
        raise ValueError('Invalid RNN type')

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=768,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AdvancedXLNetClassifier(
        config_path=ARGS.xlnet_config_path,
        model_path=ARGS.xlnet_model_path,
        encoder=encoder,
        vocab=vocab,
        encoder_dropout=0.5,
        train_xlnet=ARGS.finetune_embeddings
    )

    return model


def build_simple_xlnet(vocabulary: Vocabulary) -> Model:
    return SimpleXLNetClassifier(
        vocab=vocabulary,
        config_path=ARGS.xlnet_config_path,
        model_path=ARGS.xlnet_model_path,
        train_xlnet=ARGS.finetune_embeddings
    )


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
        embedding_dropout=ARGS.RNN_DROPOUT
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

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
        relation_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        relation_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    document_relation_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_encoder.get_output_dim(),
        model_dim=ARGS.TRANSFORMER_DIM,
        feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
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
        encoder_dropout=ARGS.RNN_DROPOUT
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

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        relation_sentence_encoder = encoder_fn(
            input_dim=embedding_dim,
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=ARGS.RNN_LAYERS,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        relation_sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    relational_encoder = RelationalTransformerEncoder(
        src_input_dim=sentence_encoder.get_output_dim(),
        kb_input_dim=relation_sentence_encoder.get_output_dim(),
        model_dim=ARGS.TRANSFORMER_DIM,
        feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
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
        encoder_dropout=ARGS.RNN_DROPOUT
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

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        sentence_encoder = transformer_seq2seq(
            input_dim=embedding_dim,
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=6,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )
        document_encoder = transformer_seq2seq(
            input_dim=sentence_encoder.get_output_dim(),
            model_dim=ARGS.TRANSFORMER_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=ARGS.TRANSFORMER_DIM,
            ttype=ARGS.WHICH_TRANSFORMER
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = HierarchicalAttentionNetwork(
        word_embeddings=word_embeddings,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
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
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the 2 RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)
    elif ARGS.ENCODER_TYPE == 'transformer':
        encoder = transformer_seq2seq(
            input_dim=hidden_dim,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AdvancedAttentionBertClassifier(
        bert_path=ARGS.BERT_PATH,
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
    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    bert = bert_embeddings(ARGS.BERT_PATH)
    embedding_dim = bert.get_output_dim()

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        sentence_encoder = encoder_fn(input_dim=embedding_dim,
                                      output_dim=ARGS.HIDDEN_DIM,
                                      num_layers=ARGS.RNN_LAYERS,
                                      bidirectional=ARGS.BIDIRECTIONAL,
                                      dropout=dropout)
        document_encoder = encoder_fn(
            input_dim=sentence_encoder.get_output_dim(),
            output_dim=ARGS.HIDDEN_DIM,
            num_layers=1,
            bidirectional=ARGS.BIDIRECTIONAL,
            dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = HierarchicalBert(
        bert_path=ARGS.BERT_PATH,
        sentence_encoder=sentence_encoder,
        document_encoder=document_encoder,
        vocab=vocab,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def build_zero_trian(vocab: Vocabulary) -> Model:
    """
    Builds the ZeroTriAN classifier without the extra features (NER, POS, HC)
    nor the relations..

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `ZeroTrian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_q_rel + 2*p_a_rel
    p_input_size = 2*embedding_dim
    # q_emb
    q_input_size = embedding_dim
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = ZeroTrian(
        word_embeddings=word_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=0,
        encoder_dropout=ARGS.RNN_DROPOUT
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
    A `Trian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    rel_embeddings = learned_embeddings(vocab, ARGS.REL_EMBEDDING_DIM,
                                        'tokens')

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_q_rel + 2*p_a_rel
    p_input_size = (2*embedding_dim + + 3*ARGS.REL_EMBEDDING_DIM)
    # q_emb
    q_input_size = embedding_dim
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
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
        encoder_dropout=ARGS.RNN_DROPOUT
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
    rel_embeddings = learned_embeddings(vocab, ARGS.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    hidden_dim = 100

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        encoder = encoder_fn(input_dim=hidden_dim,
                             output_dim=ARGS.HIDDEN_DIM,
                             num_layers=ARGS.RNN_LAYERS,
                             bidirectional=ARGS.BIDIRECTIONAL,
                             dropout=dropout)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AdvancedBertClassifier(
        bert_path=ARGS.BERT_PATH,
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
    model = SimpleBertClassifier(bert_path=ARGS.BERT_PATH, vocab=vocab)
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

    if ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        encoder_fn = lstm_encoder

    embedding_dim = embeddings.get_output_dim()
    encoder = encoder_fn(embedding_dim, ARGS.HIDDEN_DIM,
                         num_layers=ARGS.RNN_LAYERS,
                         bidirectional=ARGS.BIDIRECTIONAL)

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
    A `AttentiveReader` model ready to be trained.
    """
    embeddings = get_word_embeddings(vocab)

    embedding_dim = embeddings.get_output_dim()
    p_encoder = gru_seq2seq(embedding_dim, ARGS.HIDDEN_DIM,
                            num_layers=ARGS.RNN_LAYERS,
                            bidirectional=ARGS.BIDIRECTIONAL)
    q_encoder = gru_encoder(embedding_dim, ARGS.HIDDEN_DIM, num_layers=1,
                            bidirectional=ARGS.BIDIRECTIONAL)
    a_encoder = gru_encoder(embedding_dim, ARGS.HIDDEN_DIM, num_layers=1,
                            bidirectional=ARGS.BIDIRECTIONAL)

    model = AttentiveReader(
        embeddings, p_encoder, q_encoder, a_encoder, vocab
    )
    return model


def build_trian(vocab: Vocabulary) -> Model:
    """
    Builds the TriAN classifier using Glove embeddings and RNN encoders.

    Parameters
    ---------
    vocab : Vocabulary built from the problem dataset.

    Returns
    -------
    A `Trian` model ready to be trained.
    """
    word_embeddings = get_word_embeddings(vocab)
    pos_embeddings = learned_embeddings(vocab, ARGS.POS_EMBEDDING_DIM,
                                        'pos_tokens')
    ner_embeddings = learned_embeddings(vocab, ARGS.NER_EMBEDDING_DIM,
                                        'ner_tokens')
    rel_embeddings = learned_embeddings(vocab, ARGS.REL_EMBEDDING_DIM,
                                        'rel_tokens')

    if ARGS.ENCODER_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif ARGS.ENCODER_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    elif ARGS.ENCODER_TYPE == 'transformer':
        # Transformer has to be handled differently, but the two RNNs can share
        pass
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_pos_emb + p_ner_emb + p_q_rel + 2*p_a_rel
    #       + hc_feat
    p_input_size = (2*embedding_dim + ARGS.POS_EMBEDDING_DIM
                    + ARGS.NER_EMBEDDING_DIM + 3*ARGS.REL_EMBEDDING_DIM
                    + ARGS.HANDCRAFTED_DIM)
    # q_emb + q_pos_emb
    q_input_size = embedding_dim + ARGS.POS_EMBEDDING_DIM
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = ARGS.RNN_DROPOUT if ARGS.RNN_LAYERS > 1 else 0

    if ARGS.ENCODER_TYPE in ['lstm', 'gru']:
        p_encoder = encoder_fn(input_dim=p_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=ARGS.RNN_LAYERS,
                               bidirectional=ARGS.BIDIRECTIONAL,
                               dropout=dropout)
        q_encoder = encoder_fn(input_dim=q_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
        a_encoder = encoder_fn(input_dim=a_input_size,
                               output_dim=ARGS.HIDDEN_DIM,
                               num_layers=1,
                               bidirectional=ARGS.BIDIRECTIONAL)
    elif ARGS.ENCODER_TYPE == 'transformer':
        p_encoder = transformer_seq2seq(
            input_dim=p_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=1024
        )
        q_encoder = transformer_seq2seq(
            input_dim=q_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )
        a_encoder = transformer_seq2seq(
            input_dim=a_input_size,
            model_dim=ARGS.HIDDEN_DIM,
            num_layers=4,
            num_attention_heads=4,
            feedforward_hidden_dim=512
        )

    # Instantiate modele with our embedding, encoder and vocabulary
    model = Trian(
        word_embeddings=word_embeddings,
        rel_embeddings=rel_embeddings,
        pos_embeddings=pos_embeddings,
        ner_embeddings=ner_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=ARGS.EMBEDDDING_DROPOUT,
        encoder_dropout=ARGS.RNN_DROPOUT
    )

    return model


def test_load(build_model_fn: Callable[[Vocabulary], Model],
              reader: BaseReader,
              save_path: Path,
              original_prediction: torch.Tensor,
              cuda_device: int) -> None:
    "Test if we can load the model and if its prediction matches the original."
    print('\n>>>>Testing if the model saves and loads correctly')
    # Reload vocabulary
    with open(save_path / 'vocabulary.pickle', 'rb') as vocab_file:
        model = build_model_fn(pickle.load(vocab_file))
    # Recreate the model.
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
        question_type="",
        passage=passage,
        question=question,
        answer0=answer0,
        answer1=answer1
    )
    np.testing.assert_array_almost_equal(
        original_prediction['logits'], prediction['logits'])
    print('Success.')


def create_reader(reader_type: str) -> BaseReader:
    "Returns the appropriate Reder instance from the type and configuration."
    is_bert = ARGS.EMBEDDING_TYPE == 'bert'
    if reader_type == 'simple':
        return SimpleMcScriptReader(embedding_type=ARGS.EMBEDDING_TYPE,
                                    max_seq_length=ARGS.max_seq_length,
                                    xlnet_vocab_file=ARGS.xlnet_vocab_path)
    if reader_type == 'full-trian':
        return FullTrianReader(is_bert=is_bert,
                               conceptnet_path=ARGS.CONCEPTNET_PATH)
    if reader_type == 'simple-bert':
        return SimpleBertReader()
    if reader_type == 'simple-trian':
        return SimpleTrianReader(embedding_type=ARGS.EMBEDDING_TYPE,
                                 max_seq_length=ARGS.max_seq_length,
                                 xlnet_vocab_file=ARGS.xlnet_vocab_path,
                                 conceptnet_path=ARGS.CONCEPTNET_PATH)
    if reader_type == 'relation-bert':
        return RelationBertReader(is_bert=is_bert,
                                  conceptnet_path=ARGS.CONCEPTNET_PATH)
    if reader_type == 'simple-xl':
        return SimpleXLNetReader(
            vocab_file=ARGS.xlnet_vocab_path,
            max_seq_length=ARGS.max_seq_length
        )
    if reader_type == 'relation-xl':
        return RelationXLNetReader(
            vocab_file=ARGS.xlnet_vocab_path,
            max_seq_length=ARGS.max_seq_length,
            conceptnet_path=ARGS.CONCEPTNET_PATH
        )
    raise ValueError(f'Reader type {reader_type} is invalid')


def get_modelfn_reader() -> Tuple[Callable[[Vocabulary], Model], BaseReader]:
    "Gets the build function and reader for the model"
    # Model -> Build function, reader type
    models = {
        'baseline': (build_baseline, 'simple'),
        'trian': (build_trian, 'full-trian'),
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
        'zero-trian': (build_zero_trian, 'simple'),
        'simple-xl': (build_simple_xlnet, 'simple-xl'),
        'advanced-xl': (build_advanced_xlnet, 'simple-xl'),
        'relation-xl': (build_relational_xl, 'relation-xl'),
    }

    if ARGS.MODEL in models:
        build_fn, reader_type = models[ARGS.MODEL]
        return build_fn, create_reader(reader_type)
    raise ValueError(f'Invalid model name: {ARGS.MODEL}')


def make_prediction(model: Model,
                    reader: BaseReader,
                    verbose: bool = False
                    ) -> torch.Tensor:
    "Create a predictor to run our model and get predictions."
    model.eval()
    predictor = McScriptPredictor(model, reader)

    if verbose:
        print()
        print('#'*5, 'EXAMPLE', '#'*5)

    passage, question, answer1, label1 = example_input(0)
    _, _, answer2, _ = example_input(1)
    result = predictor.predict("", "", "", passage, question, answer1, answer2)
    prediction = np.argmax(result['prob'])

    if verbose:
        print('Passage:\n', '\t', passage, sep='')
        print('Question:\n', '\t', question, sep='')
        print('Answers:')
        print('\t1:', answer1)
        print('\t2:', answer2)
        print('Prediction:', prediction+1)
        print('Correct:', 1 if label1 == 1 else 2)

    return result


def split_list(data: List[Instance]) -> Dict[str, List[Instance]]:
    output: Dict[str, List[Instance]] = defaultdict(list)

    for sample in data:
        qtype = sample['metadata']['question_type']
        output[qtype].append(sample)

    return output


def evaluate(model: Model,
             reader: BaseReader,
             test_data: List[Instance]
             ) -> None:
    vocab = Vocabulary.from_instances(test_data)
    iterator = BucketIterator(batch_size=ARGS.BATCH_SIZE,
                              sorting_keys=reader.keys)
    # Our data should be indexed using the vocabulary we learned.
    iterator.index_with(vocab)

    data_types = split_list(test_data)
    results: Dict[str, Tuple[int, float]] = {}

    print()
    print('#'*5, 'PER TYPE EVALUATION', '#'*5)
    for qtype, data in data_types.items():
        num_items = len(data)
        print(f'Type: {qtype} ({num_items})')

        metrics = allen_eval(model, data, iterator, ARGS.CUDA_DEVICE, "")
        print()

        accuracy = metrics['accuracy']
        results[qtype] = (num_items, accuracy)


def run_model() -> None:
    "Execute model according to the configuration"
    # Which model to use?
    build_fn, reader = get_modelfn_reader()

    def optimiser(model: Model) -> torch.optim.Optimizer:
        return Adamax(model.parameters(), lr=2e-3)

    # Create SAVE_FOLDER if it doesn't exist
    ARGS.SAVE_FOLDER.mkdir(exist_ok=True, parents=True)
    train_dataset = load_data(data_path=ARGS.TRAIN_DATA_PATH,
                              reader=reader,
                              pre_processed_path=ARGS.TRAIN_PREPROCESSED_PATH)
    val_dataset = load_data(data_path=ARGS.VAL_DATA_PATH,
                            reader=reader,
                            pre_processed_path=ARGS.VAL_PREPROCESSED_PATH)
    test_dataset = load_data(data_path=ARGS.TEST_DATA_PATH,
                             reader=reader,
                             pre_processed_path=ARGS.TEST_PREPROCESSED_PATH)

    model = train_model(build_fn,
                        train_data=train_dataset,
                        val_data=val_dataset,
                        test_data=test_dataset,
                        save_path=ARGS.SAVE_PATH,
                        num_epochs=ARGS.NUM_EPOCHS,
                        batch_size=ARGS.BATCH_SIZE,
                        optimiser_fn=optimiser,
                        cuda_device=ARGS.CUDA_DEVICE,
                        sorting_keys=reader.keys)

    evaluate(model, test_dataset, reader)
    result = make_prediction(model, reader, verbose=False)

    print('Save path', ARGS.SAVE_PATH)

    cuda_device = 0 if is_cuda(model) else -1
    test_load(build_fn, reader, ARGS.SAVE_PATH, result, cuda_device)


if __name__ == '__main__':
    torch.manual_seed(ARGS.RANDOM_SEED)

    run_model()
