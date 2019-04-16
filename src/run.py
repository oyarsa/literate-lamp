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
from typing import Dict
import sys
import os

import torch
import numpy as np

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
from torch.optim import Adam, Adamax, Adagrad, SGD

from models import BaselineClassifier, AttentiveClassifier, AttentiveReader
from predictor import McScriptPredictor
from reader import McScriptReader
from util import (example_input, is_cuda, train_model, glove_embeddings,
                  lstm_encoder, gru_encoder, lstm_seq2seq, gru_seq2seq,
                  get_experiment_name, learned_embeddings)

DEFAULT_CONFIG = 'medium'  # Can be: _medium_ , _large_ or _small_
CONFIG = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONFIG

# Which model to use: 'baseline' or 'attentive' for now.
DEFAULT_MODEL = 'attentive'
MODEL = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL

# Which optimiser to use: 'adam', 'adamax', 'adagrad', 'sgd'
DEFAULT_OPTIM = 'adam'
OPTIMISER = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_OPTIM

# Whether to fine-tune embeddings or not
DEFAULT_FINETUNE = True
FINETUNE = sys.argv[4] == 'True' if len(sys.argv) >= 5 else DEFAULT_FINETUNE

NUMBER_EPOCHS = int(sys.argv[5]) if len(sys.argv) >= 6 else None
RNN_HIDDEN_SIZE = int(sys.argv[6]) if len(sys.argv) >= 7 else None
EMBEDDING_SIZE = int(sys.argv[7]) if len(sys.argv) >= 8 else None
NER_EMBEDDING_DIM = 8
POS_EMBEDDING_DIM = 12

# TODO: Proper configuration path for the External folder. The data one is
# going to be part of the repo, so this is fine for now, but External isn't
# always going to be.
if CONFIG == 'large':
    # Path to our dataset
    DATA_PATH = './data/mctrain-data.json'
    if EMBEDDING_SIZE is None or EMBEDDING_SIZE == 300:
        # Path to our embeddings
        GLOVE_PATH = '../External/glove.840B.300d.txt'
        # Size of our embeddings
        EMBEDDING_DIM = 300
    elif EMBEDDING_SIZE is not None and EMBEDDING_SIZE == 100:
        # Path to our embeddings
        GLOVE_PATH = '../External/glove.6B.100d.txt'
        # Size of our embeddings
        EMBEDDING_DIM = 100
    else:
        raise ValueError('Invalid embedding size')
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = RNN_HIDDEN_SIZE or 96
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/data.processed.pickle'
    # Size of minibatch
    BATCH_SIZE = 32
    # Number of epochs to train model
    NUM_EPOCHS = NUMBER_EPOCHS or 1000
elif CONFIG == 'small':
    # Path to our dataset
    DATA_PATH = './data/small.json'
    # Path to our embeddings
    GLOVE_PATH = '../External/glove.6B.50d.txt'
    # Size of our embeddings
    EMBEDDING_DIM = 50
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 50
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/small.processed.pickle'
    # Size of minibatch
    BATCH_SIZE = 3
    # Number of epochs to train model
    NUM_EPOCHS = NUMBER_EPOCHS or 5
elif CONFIG == 'medium':
    # Path to our dataset
    DATA_PATH = './data/mcdev-data.json'
    # Path to our embeddings
    GLOVE_PATH = '../External/glove.6B.100d.txt'
    # Size of our embeddings
    EMBEDDING_DIM = 100
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 64
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/medium.processed.pickle'
    # Size of minibatch
    BATCH_SIZE = 25
    # Number of epochs to train model
    NUM_EPOCHS = NUMBER_EPOCHS or 10

# Path to save the Model and Vocabulary
SAVE_FOLDER = './experiments/'
SAVE_PATH = SAVE_FOLDER + get_experiment_name(MODEL, CONFIG) \
    + f'-{OPTIMISER}-{FINETUNE}-{RNN_HIDDEN_SIZE}-{EMBEDDING_SIZE}' + '/'
print('Save path', SAVE_PATH)
# Random seed (for reproducibility)
RANDOM_SEED = 1234

# Model Configuration
# Use LSTM or GRU
RNN_TYPE = 'lstm'
BIDIRECTIONAL = True
RNN_LAYERS = 2
RNN_DROPOUT = 0
EMBEDDDING_DROPOUT = 0.5


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
    embeddings = glove_embeddings(vocab, GLOVE_PATH, EMBEDDING_DIM)
    if RNN_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif RNN_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    encoder = encoder_fn(EMBEDDING_DIM, HIDDEN_DIM, num_layers=RNN_LAYERS,
                         bidirectional=BIDIRECTIONAL)

    # Instantiate modele with our embedding, encoder and vocabulary
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
    embeddings = glove_embeddings(vocab, GLOVE_PATH, EMBEDDING_DIM,
                                  training=True)

    p_encoder = gru_seq2seq(EMBEDDING_DIM, HIDDEN_DIM, num_layers=RNN_LAYERS,
                            bidirectional=BIDIRECTIONAL)
    q_encoder = gru_encoder(EMBEDDING_DIM, HIDDEN_DIM, num_layers=1,
                            bidirectional=BIDIRECTIONAL)
    a_encoder = gru_encoder(EMBEDDING_DIM, HIDDEN_DIM, num_layers=1,
                            bidirectional=BIDIRECTIONAL)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AttentiveReader(
        embeddings, p_encoder, q_encoder, a_encoder, vocab)
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
    word_embeddings = glove_embeddings(vocab, GLOVE_PATH, EMBEDDING_DIM,
                                       training=FINETUNE)
    pos_embeddings = learned_embeddings(vocab, POS_EMBEDDING_DIM, 'pos_tokens')
    ner_embeddings = learned_embeddings(vocab, NER_EMBEDDING_DIM, 'ner_tokens')

    if RNN_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif RNN_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    else:
        raise ValueError('Invalid RNN type')

    # p_emb + p_q_weighted + p_pos_emb + p_ner_emb
    p_input_size = 2*EMBEDDING_DIM + POS_EMBEDDING_DIM + NER_EMBEDDING_DIM
    # q_emb + q_pos_emb
    q_input_size = EMBEDDING_DIM + POS_EMBEDDING_DIM
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * EMBEDDING_DIM

    p_encoder = encoder_fn(input_dim=p_input_size, output_dim=HIDDEN_DIM,
                           num_layers=RNN_LAYERS, bidirectional=BIDIRECTIONAL,
                           dropout=RNN_DROPOUT)
    q_encoder = encoder_fn(input_dim=q_input_size, output_dim=HIDDEN_DIM,
                           num_layers=1, bidirectional=BIDIRECTIONAL,
                           dropout=RNN_DROPOUT)
    a_encoder = encoder_fn(input_dim=a_input_size, output_dim=HIDDEN_DIM,
                           num_layers=1, bidirectional=BIDIRECTIONAL,
                           dropout=RNN_DROPOUT)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = AttentiveClassifier(
        word_embeddings=word_embeddings,
        pos_embeddings=pos_embeddings,
        ner_embeddings=ner_embeddings,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        a_encoder=a_encoder,
        vocab=vocab,
        embedding_dropout=0.4,
        encoder_dropout=0.4
    )

    return model


def test_attentive_reader_load(save_path: str,
                               original_prediction: Dict[str, torch.Tensor],
                               embeddings: TextFieldEmbedder,
                               p_encoder: Seq2VecEncoder,
                               q_encoder: Seq2VecEncoder,
                               a_encoder: Seq2VecEncoder,
                               cuda_device: int) -> None:
    """
    Test if we can load the model and if its prediction matches the original.

    Parameters
    ----------
    save_path : Path to the folder where the model was saved.
    original_prediction : The prediction from the model for `example_input`
        before it was saved.  embeddings : The Embedding layer used for the
        model.  encoder : The Encoder layer used for the model.
    cuda_device: Device number. -1 if CPU, >= 0 if GPU.
    """
    # Reload vocabulary
    vocab = Vocabulary.from_files(save_path + 'vocabulary')
    # Recreate the model.
    model = AttentiveReader(
        embeddings, p_encoder, q_encoder, a_encoder, vocab)
    # Load the state from the file
    with open(save_path + 'model.th', 'rb') as f:
        model.load_state_dict(torch.load(f))
    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    # Try predicting again and see if we get the same results (we should).
    predictor = McScriptPredictor(model, dataset_reader=McScriptReader())
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
        original_prediction['prob'], prediction['prob'])


def test_attentive_load(save_path: str,
                        original_prediction: Dict[str, torch.Tensor],
                        word_embeddings: TextFieldEmbedder,
                        pos_embeddings: TextFieldEmbedder,
                        ner_embeddings: TextFieldEmbedder,
                        p_encoder: Seq2VecEncoder,
                        q_encoder: Seq2VecEncoder,
                        a_encoder: Seq2VecEncoder,
                        cuda_device: int) -> None:
    """
    Test if we can load the model and if its prediction matches the original.

    Parameters
    ----------
    save_path : Path to the folder where the model was saved.
    original_prediction : The prediction from the model for `example_input`
        before it was saved.  embeddings : The Embedding layer used for the
        model.  encoder : The Encoder layer used for the model.
    cuda_device: Device number. -1 if CPU, >= 0 if GPU.
    """
    # Reload vocabulary
    vocab = Vocabulary.from_files(save_path + 'vocabulary')
    # Recreate the model.
    model = AttentiveClassifier(
        word_embeddings, pos_embeddings, ner_embeddings, p_encoder, q_encoder,
        a_encoder, vocab)
    # Load the state from the file
    with open(save_path + 'model.th', 'rb') as f:
        model.load_state_dict(torch.load(f))
    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    # Try predicting again and see if we get the same results (we should).
    predictor = McScriptPredictor(model, dataset_reader=McScriptReader())
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


def test_baseline_load(save_path: str,
                       original_prediction: Dict[str, torch.Tensor],
                       embeddings: TextFieldEmbedder,
                       encoder: Seq2VecEncoder,
                       cuda_device: int) -> None:
    """
    Test if we can load the model and if its prediction matches the original.

    Parameters
    ----------
    save_path : Path to the folder where the model was saved.
    original_prediction : The prediction from the model for `example_input`
        before it was saved.  embeddings : The Embedding layer used for the
        model.  encoder : The Encoder layer used for the model.
    cuda_device: Device number. -1 if CPU, >= 0 if GPU.
    """
    # Reload vocabulary
    vocab = Vocabulary.from_files(save_path + 'vocabulary')
    # Recreate the model.
    model = BaselineClassifier(embeddings, encoder, vocab)
    # Load the state from the file
    with open(save_path + 'model.th', 'rb') as f:
        model.load_state_dict(torch.load(f))
    # We've loaded the model. Let's move it to the GPU again if available.
    if cuda_device > -1:
        model.cuda(cuda_device)

    # Try predicting again and see if we get the same results (we should).
    predictor = McScriptPredictor(model, dataset_reader=McScriptReader())
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


def run_model() -> None:
    # Which model to use?
    if MODEL == 'baseline':
        build_fn = build_baseline
    elif MODEL == 'attentive':
        build_fn = build_attentive
    elif MODEL == 'reader':
        build_fn = build_attentive_reader
    else:
        raise ValueError('Invalid model name')

    # Train and save our model
    def optimiser(model: Model) -> torch.optim.Optimizer:
        if OPTIMISER == 'adam':
            return Adam(model.parameters(), lr=0.001)
        elif OPTIMISER == 'adamax':
            return Adamax(model.parameters())
        elif OPTIMISER == 'adagrad':
            return Adagrad(model.parameters())
        elif OPTIMISER == 'sgd':
            return SGD(model.parameters(), lr=0.2, momentum=0.9, nesterov=True)
        else:
            raise ValueError('Invalid optimiser')

    # Create SAVE_FOLDER if it doesn't exist
    if not os.path.exists(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    model = train_model(build_fn, data_path=DATA_PATH,
                        save_path=SAVE_PATH, num_epochs=NUM_EPOCHS,
                        patience=50, batch_size=BATCH_SIZE,
                        pre_processed_path=PREPROCESSED_PATH,
                        optimiser_fn=optimiser)

    # Create a predictor to run our model and get predictions.
    reader = McScriptReader()
    predictor = McScriptPredictor(model, reader)

    print()
    print('#'*5, 'EXAMPLE',  '#'*5)
    passage, question, answer1, label1 = example_input(0)
    _, _, answer2, label2 = example_input(1)
    result = predictor.predict("", "", passage, question, answer1, answer2)
    prediction = np.argmax(result['prob'])

    print('Passage:\n', '\t', passage, sep='')
    print('Question:\n', '\t', question, sep='')
    print('Answers:')
    print('\t1:', answer1)
    print('\t2:', answer2)
    print('Prediction:', prediction+1)
    print('Correct:', 1 if label1 == 1 else 2)

    # Test if we can load the saved model
    cuda_device = 0 if is_cuda(model) else -1
    if MODEL == 'baseline':
        test_baseline_load(SAVE_PATH, result,
                           model.word_embeddings, model.q_encoder, cuda_device)
    elif MODEL == 'attentive':
        test_attentive_load(SAVE_PATH, result,
                            model.word_embeddings, model.pos_embeddings,
                            model.ner_embeddings, model.p_encoder,
                            model.q_encoder, model.a_encoder, cuda_device)
    elif MODEL == 'reader':
        test_attentive_reader_load(SAVE_PATH, result,
                                   model.word_embeddings, model.p_encoder,
                                   model.q_encoder, model.a_encoder,
                                   cuda_device)
    else:
        raise ValueError('Invalid model name')


if __name__ == '__main__':
    # Manual seeding for reproducibility.
    torch.manual_seed(RANDOM_SEED)

    run_model()
