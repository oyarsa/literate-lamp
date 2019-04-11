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

import torch
import numpy as np

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from models import BaselineClassifier
from predictor import McScriptPredictor
from reader import McScriptReader
from util import (example_input, is_cuda, train_model, glove_embeddings,
                  lstm_encoder, gru_encoder)

DEFAULT_CONFIG = 'medium'  # Can be: _medium_ , _large_ or _small_
CONFIG = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_CONFIG

# TODO: Proper configuration path for the External folder. The data one is
# going to be part of the repo, so this is fine for now, but External isn't
# always going to be.
if CONFIG == 'large':
    # Path to our dataset
    DATA_PATH = './data/data.csv'
    # Path to our embeddings
    GLOVE_PATH = '../External/glove.840B.300d.txt'
    # Size of our embeddings
    EMBEDDING_DIM = 300
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 96
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/data.processed.pickle'
elif CONFIG == 'small':
    # Path to our dataset
    DATA_PATH = './data/small.csv'
    # Path to our embeddings
    GLOVE_PATH = '../External/glove.6B.50d.txt'
    # Size of our embeddings
    EMBEDDING_DIM = 50
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 50
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/small.processed.pickle'
elif CONFIG == 'medium':
    # Path to our dataset
    DATA_PATH = './data/medium.csv'
    # Path to our embeddings
    GLOVE_PATH = '../External/glove.6B.100d.txt'
    # Size of our embeddings
    EMBEDDING_DIM = 100
    # Size of our hidden layers (for each encoder)
    HIDDEN_DIM = 64
    # Path to save pre-processed input
    PREPROCESSED_PATH = '../External/medium.processed.pickle'

# Number of epochs to train model
NUM_EPOCHS = 10
# Path to save the Model and Vocabulary
SAVE_PATH = "/tmp/"
# Random seed (for reproducibility)
RANDOM_SEED = 1234
# Size of minibatch
BATCH_SIZE = 128

# Model Configuration
# Use LSTM or GRU
RNN_TYPE = 'lstm'
BIDIRECTIONAL = True
RNN_LAYERS = 2


def build_baseline(vocab: Vocabulary) -> Model:
    """
    Builds the Baseline classifier using Glove embeddings and LSTM encoders.

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


def test_load(save_path: str,
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
    passage, question, answer, _ = example_input(1)
    prediction = predictor.predict(
        passage=passage,
        question=question,
        answer=answer
    )
    np.testing.assert_array_almost_equal(
        original_prediction['logits'], prediction['logits'])


if __name__ == '__main__':
    # Manual seeding for reproducibility.
    torch.manual_seed(RANDOM_SEED)

    # Train and save our model
    model = train_model(build_baseline, data_path=DATA_PATH,
                        save_path=SAVE_PATH, num_epochs=NUM_EPOCHS,
                        patience=50, batch_size=BATCH_SIZE,
                        pre_processed_path=PREPROCESSED_PATH)

    # Create a predictor to run our model and get predictions.
    reader = McScriptReader()
    predictor = McScriptPredictor(model, dataset_reader=reader)
    for index in [0, 1]:
        print('#'*5, 'EXAMPLE', index, '#'*5)
        # Execute prediction. Gets output dict from the model.
        passage, question, answer, label = example_input(index)
        prediction = predictor.predict(
            passage=passage,
            question=question,
            answer=answer
        )
        instance = reader.text_to_instance(passage, question, answer, label)
        print(instance)
        # Predicted class
        prob = prediction['prob']
        print('\t Predicted:', prob)
        print()

    cuda_device = 0 if is_cuda(model) else -1
    # Test if we can load the saved model
    test_load(SAVE_PATH, prediction, model.word_embeddings,
              model.q_encoder, cuda_device)
