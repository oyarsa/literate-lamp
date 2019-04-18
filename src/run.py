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
from typing import Dict, Callable
import sys
from pathlib import Path
import pickle

import torch
from torch.optim import Adamax
import numpy as np

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from models import BaselineClassifier, AttentiveClassifier, AttentiveReader
from predictor import McScriptPredictor
from util import (example_input, is_cuda, train_model, get_experiment_name,
                  load_data, create_reader)
from layers import (lstm_encoder, gru_encoder, lstm_seq2seq, gru_seq2seq,
                    glove_embeddings, learned_embeddings, bert_embeddings)

DEFAULT_CONFIG = 'medium'  # Can be: _medium_ , _large_ or _small_
CONFIG = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONFIG

# Which model to use: 'baseline' or 'attentive' for now.
DEFAULT_MODEL = 'attentive'
MODEL = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_MODEL

NER_EMBEDDING_DIM = 8
REL_EMBEDDING_DIM = 10
POS_EMBEDDING_DIM = 12
HANDCRAFTED_DIM = 7
DEFAULT_EMBEDDING_TYPE = 'glove'  # can also be 'glove'
EMBEDDING_TYPE = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_EMBEDDING_TYPE

CUDA_DEVICE = int(sys.argv[4]) if len(sys.argv) >= 5 else 0

USAGE = """
USAGE:
    run.py CONFIG MODEL [EMBEDDING_TYPE] [CUDA_DEVICE]

ARGS:
    CONFIG: configuration to use. One of: small, medium, large
    MODEL: model to run. One of: baseline, attentive, reader
    EMBEDDING_TYPE: word embeddings for the text. One of: glove, bert.
    CUDA_DEVICE: device to run the training. -1 for CPU, >=0 for GPU.
"""

DATA_FOLDER = Path('data')
EXTERNAL_FOLDER = Path('..', 'External')

# TODO: Proper configuration path for the External folder. The data one is
# going to be part of the repo, so this is fine for now, but External isn't
# always going to be.
if CONFIG == 'large':
    # Path to our dataset
    TRAIN_DATA_PATH = DATA_FOLDER / 'mctrain-data.json'
    VAL_DATA_PATH = DATA_FOLDER / 'mcdev-data.json'
    TEST_DATA_PATH = DATA_FOLDER / 'mctest-data.json'
    # Path to our embeddings
    GLOVE_PATH = EXTERNAL_FOLDER / 'glove.840B.300d.txt'
    # Size of our embeddings
    GLOVE_EMBEDDING_DIM = 300
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 96
    # Size of minibatch
    BATCH_SIZE = 32
    # Number of epochs to train model
    NUM_EPOCHS = 50
elif CONFIG == 'small':
    # Path to our dataset
    TRAIN_DATA_PATH = DATA_FOLDER / 'small-train.json'
    VAL_DATA_PATH = DATA_FOLDER / 'small-dev.json'
    TEST_DATA_PATH = DATA_FOLDER / 'small-test.json'
    # Path to our embeddings
    GLOVE_PATH = EXTERNAL_FOLDER / 'glove.6B.50d.txt'
    # Size of our embeddings
    GLOVE_EMBEDDING_DIM = 50
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 50
    # Size of minibatch
    BATCH_SIZE = 3
    # Number of epochs to train model
    NUM_EPOCHS = 5
elif CONFIG == 'medium':
    # Path to our dataset
    DATA_PATH = DATA_FOLDER / 'mcdev-data.json'
    # Path to our embeddings
    GLOVE_PATH = EXTERNAL_FOLDER / 'glove.6B.100d.txt'
    # Size of our embeddings
    GLOVE_EMBEDDING_DIM = 100
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 64
    # Size of minibatch
    BATCH_SIZE = 25
    # Number of epochs to train model
    NUM_EPOCHS = 10

BERT_PATH = EXTERNAL_FOLDER / 'bert-base-uncased.tar.gz'
CONCEPTNET_PATH = EXTERNAL_FOLDER / 'conceptnet.csv'

# Path to save the Model and Vocabulary
SAVE_FOLDER = Path('experiments')
SAVE_PATH = SAVE_FOLDER / get_experiment_name(MODEL, CONFIG)
print('Save path', SAVE_PATH)

# Path to save pre-processed input
TRAIN_PREPROCESSED_NAME = f'train.{CONFIG}.{EMBEDDING_TYPE}.processed.pickle'
VAL_PREPROCESSED_NAME = f'val.{CONFIG}.{EMBEDDING_TYPE}.processed.pickle'
TEST_PREPROCESSED_NAME = f'test.{CONFIG}.{EMBEDDING_TYPE}.processed.pickle'

TRAIN_PREPROCESSED_PATH = EXTERNAL_FOLDER / TRAIN_PREPROCESSED_NAME
VAL_PREPROCESSED_PATH = EXTERNAL_FOLDER / VAL_PREPROCESSED_NAME
TEST_PREPROCESSED_PATH = EXTERNAL_FOLDER / TEST_PREPROCESSED_NAME
print('Pre-processed data path:', TRAIN_PREPROCESSED_PATH)

# Random seed (for reproducibility)
RANDOM_SEED = 1234

# Model Configuration
# Use LSTM or GRU
RNN_TYPE = 'lstm'
BIDIRECTIONAL = True
RNN_LAYERS = 1
RNN_DROPOUT = 0.5
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
    if EMBEDDING_TYPE == 'glove':
        embeddings = glove_embeddings(vocab, GLOVE_PATH, GLOVE_EMBEDDING_DIM,
                                      training=True)
    elif EMBEDDING_TYPE == 'bert':
        embeddings = bert_embeddings(pretrained_model=BERT_PATH)
    else:
        raise ValueError('Invalid word embedding type')

    if RNN_TYPE == 'lstm':
        encoder_fn = lstm_encoder
    elif RNN_TYPE == 'gru':
        encoder_fn = gru_encoder
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = embeddings.get_output_dim()
    encoder = encoder_fn(embedding_dim, HIDDEN_DIM, num_layers=RNN_LAYERS,
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
    if EMBEDDING_TYPE == 'glove':
        embeddings = glove_embeddings(vocab, GLOVE_PATH, GLOVE_EMBEDDING_DIM,
                                      training=True)
    elif EMBEDDING_TYPE == 'bert':
        embeddings = bert_embeddings(pretrained_model=BERT_PATH)
    else:
        raise ValueError('Invalid word embedding type')

    embedding_dim = embeddings.get_output_dim()
    p_encoder = gru_seq2seq(embedding_dim, HIDDEN_DIM, num_layers=RNN_LAYERS,
                            bidirectional=BIDIRECTIONAL)
    q_encoder = gru_encoder(embedding_dim, HIDDEN_DIM, num_layers=1,
                            bidirectional=BIDIRECTIONAL)
    a_encoder = gru_encoder(embedding_dim, HIDDEN_DIM, num_layers=1,
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
    if EMBEDDING_TYPE == 'glove':
        word_embeddings = glove_embeddings(vocab, GLOVE_PATH,
                                           GLOVE_EMBEDDING_DIM, training=True)
    elif EMBEDDING_TYPE == 'bert':
        word_embeddings = bert_embeddings(pretrained_model=BERT_PATH)
    else:
        raise ValueError('Invalid word embedding type')
    pos_embeddings = learned_embeddings(vocab, POS_EMBEDDING_DIM, 'pos_tokens')
    ner_embeddings = learned_embeddings(vocab, NER_EMBEDDING_DIM, 'ner_tokens')
    rel_embeddings = learned_embeddings(vocab, REL_EMBEDDING_DIM, 'rel_tokens')

    if RNN_TYPE == 'lstm':
        encoder_fn = lstm_seq2seq
    elif RNN_TYPE == 'gru':
        encoder_fn = gru_seq2seq
    else:
        raise ValueError('Invalid RNN type')

    embedding_dim = word_embeddings.get_output_dim()

    # p_emb + p_q_weighted + p_pos_emb + p_ner_emb + p_q_rel + 2*p_a_rel
    #       + hc_feat
    p_input_size = (2*embedding_dim + POS_EMBEDDING_DIM + NER_EMBEDDING_DIM
                    + 3*REL_EMBEDDING_DIM + HANDCRAFTED_DIM)
    # q_emb + q_pos_emb
    q_input_size = embedding_dim + POS_EMBEDDING_DIM
    # a_emb + a_q_match + a_p_match
    a_input_size = 3 * embedding_dim

    # To prevent the warning on single-layer, as the dropout is only
    # between layers of the stacked RNN.
    dropout = RNN_DROPOUT if RNN_LAYERS > 1 else 0

    p_encoder = encoder_fn(input_dim=p_input_size, output_dim=HIDDEN_DIM,
                           num_layers=RNN_LAYERS, bidirectional=BIDIRECTIONAL,
                           dropout=dropout)
    q_encoder = encoder_fn(input_dim=q_input_size, output_dim=HIDDEN_DIM,
                           num_layers=1, bidirectional=BIDIRECTIONAL)
    a_encoder = encoder_fn(input_dim=a_input_size, output_dim=HIDDEN_DIM,
                           num_layers=1, bidirectional=BIDIRECTIONAL)

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
        embedding_dropout=EMBEDDDING_DROPOUT,
        encoder_dropout=RNN_DROPOUT
    )

    return model


def test_load(build_model_fn: Callable[[Vocabulary], Model],
              save_path: Path,
              original_prediction: Dict[str, torch.Tensor],
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
    reader = create_reader(conceptnet_path=CONCEPTNET_PATH,
                           embedding_type=EMBEDDING_TYPE)
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


def run_model() -> None:
    "Execute model according to the configuration"
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
        return Adamax(model.parameters(), lr=2e-3)

    # Create SAVE_FOLDER if it doesn't exist
    SAVE_FOLDER.mkdir(exist_ok=True, parents=True)
    reader = create_reader(conceptnet_path=CONCEPTNET_PATH,
                           embedding_type=EMBEDDING_TYPE)
    train_dataset = load_data(data_path=TRAIN_DATA_PATH,
                              reader=reader,
                              pre_processed_path=TRAIN_PREPROCESSED_PATH)
    val_dataset = load_data(data_path=VAL_DATA_PATH,
                            reader=reader,
                            pre_processed_path=VAL_PREPROCESSED_PATH)
    test_dataset = load_data(data_path=TEST_DATA_PATH,
                             reader=reader,
                             pre_processed_path=TEST_PREPROCESSED_PATH)
    model = train_model(build_fn,
                        train_data=train_dataset,
                        val_data=val_dataset,
                        test_data=test_dataset,
                        save_path=SAVE_PATH,
                        num_epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        optimiser_fn=optimiser,
                        cuda_device=CUDA_DEVICE)

    model.eval()
    # Create a predictor to run our model and get predictions.
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

    # Test if we can load the saved model
    cuda_device = 0 if is_cuda(model) else -1
    test_load(build_fn, SAVE_PATH, result, cuda_device)


if __name__ == '__main__':
    if any('help' in arg or '-h' in arg for arg in sys.argv):
        print(USAGE)
        exit(0)
    # Manual seeding for reproducibility.
    torch.manual_seed(RANDOM_SEED)

    run_model()
