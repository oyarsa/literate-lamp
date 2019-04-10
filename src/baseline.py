#!/usr/bin/env python3
from typing import Dict

import torch
import numpy as np

# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import (BasicTextFieldEmbedder,
                                                   TextFieldEmbedder)
# This is the actual neural layer for the embedding. This will be passed into
# the embedder above.
from allennlp.modules.token_embedders import Embedding
# `Seq2VecEncoder` is an abstract encoder that takes a sequence and generates
# a vector. This can be an LSTM (although they can also be Seq2Seq if you
# output the hidden state), a Transformer or anything else really, just taking
# NxM -> 1xQ.
# The `PytorchSeq2VecWrapper` is a wrapper for the PyTorch Seq2Vec encoders
# (such as the LSTM we'll use later on), as they don't exactly follow the
# interface the library expects.
from allennlp.modules.seq2vec_encoders import (Seq2VecEncoder,
                                               PytorchSeq2VecWrapper)

from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary

from models import BaselineClassifier
from predictor import QaPredictor
from reader import QaDatasetReader
from util import example_input, is_cuda, train_model


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


def build_model(vocab: Vocabulary) -> Model:
    # Pre-trained embeddings using GloVe
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM,
                                trainable=False,
                                pretrained_file=GLOVE_PATH)
    # TODO: Not exactly sure how this one works
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    # Our encoder is going to be an LSTM. We have to wrap it for AllenNLP,
    # though.
    lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(
        EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

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
    predictor = QaPredictor(model, dataset_reader=QaDatasetReader())
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
    model = train_model(build_model, data_path=DATA_PATH, save_path=SAVE_PATH)

    # Create a predictor to run our model and get predictions.
    predictor = QaPredictor(model, dataset_reader=QaDatasetReader())
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
