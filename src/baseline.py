#!/usr/bin/env python3
from typing import Dict

import torch
import numpy as np

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from models import BaselineClassifier
from predictor import McScriptPredictor
from reader import McScriptReader
from util import (example_input, is_cuda, train_model,
                  glove_embeddings, lstm_encoder)


# Path to our dataset
DATA_PATH = './data/small.csv'
# Path to our embeddings
GLOVE_PATH = '../External/glove.840B.300d.txt'
# Size of our embeddings
EMBEDDING_DIM = 300
# Size of our hidden layers (for each encoder)
HIDDEN_DIM = 100
# Path to save the Model and Vocabulary
SAVE_PATH = "/tmp/"


def build_baseline(vocab: Vocabulary) -> Model:
    word_embeddings = glove_embeddings(vocab, GLOVE_PATH, EMBEDDING_DIM)
    lstm = lstm_encoder(EMBEDDING_DIM, HIDDEN_DIM)

    # Instantiate modele with our embedding, encoder and vocabulary
    model = BaselineClassifier(word_embeddings, lstm, vocab)
    return model


def test_load(save_path: str,
              original_prediction: Dict[str, torch.Tensor],
              embeddings: TextFieldEmbedder,
              encoder: Seq2VecEncoder,
              cuda_device: int) -> None:
    "Test if we can load the model and if its prediction matches the original."
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
    passage, question, answer, _ = example_input()
    prediction = predictor.predict(
        passage=passage,
        question=question,
        answer=answer
    )
    np.testing.assert_array_almost_equal(
        original_prediction['logits'], prediction['logits'])


if __name__ == '__main__':
    # Manual seeding for reproducibility.
    torch.manual_seed(1)

    # Train and save our model
    model = train_model(build_baseline, data_path=DATA_PATH,
                        save_path=SAVE_PATH)

    # Create a predictor to run our model and get predictions.
    predictor = McScriptPredictor(model, dataset_reader=McScriptReader())
    # Execute prediction. Gets output dict from the model.
    passage, question, answer, label = example_input()
    prediction = predictor.predict(
        passage=passage,
        question=question,
        answer=answer
    )
    # Predicted class
    class_ = prediction['class']
    print('Label:', label, '-- Predicted:', class_)

    cuda_device = 0 if is_cuda(model) else -1
    # Test if we can load the saved model
    test_load(SAVE_PATH, prediction, model.word_embeddings,
              model.q_encoder, cuda_device)
