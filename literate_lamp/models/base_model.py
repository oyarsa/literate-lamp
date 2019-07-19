"Base model for the classifiers. Responsible for handling the metrics."
from typing import Dict, Optional
from pathlib import Path
import copy

import torch

# Base class for the Model we'll implement. Inherits from `torch.nn.Model`,
# but compatible with what the rest of the AllenNLP library expects.
from allennlp.models import Model

# These create text embeddings from a `TextField` input. Since our text data
# is represented using `TextField`s, this makes sense.
# Again, `TextFieldEmbedder` is the abstract class, `BasicTextFieldEmbedder`
# is the implementation (as we're just using simple embeddings, no fancy
# ELMo or BERT so far).
from allennlp.modules.text_field_embedders import TextFieldEmbedder

# `Seq2VecEncoder` is an abstract encoder that takes a sequence and generates
# a vector. This can be an LSTM (although they can also be Seq2Seq if you
# output the hidden state), a Transformer or anything else really, just taking
# NxM -> 1xQ.
# The `PytorchSeq2VecWrapper` is a wrapper for the PyTorch Seq2Vec encoders
# (such as the LSTM we'll use later on), as they don't exactly follow the
# interface the library expects.
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, BertPooler
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.attention import Attention
from allennlp.modules.layer_norm import LayerNorm

# Holds the vocabulary, learned from the whole data. Also knows the mapping
# from the `TokenIndexer`, mapping the `Token` to an index in the vocabulary
# and vice-versa.
from allennlp.data.vocabulary import Vocabulary

# Some utilities provided by AllenNLP.
#   - `get_text_field_mask` masks the inputs according to the padding.
#   - `clone` creates N copies of a layer.
from allennlp.nn import util

from layers import (SequenceAttention, BilinearAttention, LinearSelfAttention,
                    bert_embeddings, LinearAttention, BilinearMatrixAttention,
                    # MultiHeadAttention,
                    # HeterogenousSequenceAttention,
                    # MultiHeadAttentionV2,
                    RelationalTransformerEncoder)


class BaseModel(Model):
    """
    Base class that sets up the vocabulary, as well as the accuracy and
    loss measures so the derived don't have to. These will never change
    unless the task changes.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
