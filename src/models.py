from typing import Dict, Optional

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
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

# Holds the vocabulary, learned from the whole data. Also knows the mapping
# from the `TokenIndexer`, mapping the `Token` to an index in the vocabulary
# and vice-versa.
from allennlp.data.vocabulary import Vocabulary

# Accuracy metric
from allennlp.training.metrics import CategoricalAccuracy

# Some utilities provided by AllenNLP.
#   - `get_text_field_mask` masks the inputs according to the padding.
#   - `clone` creates N copies of a layer.
from allennlp.nn import util


@Model.register('baseline-classifier')
class BaselineClassifier(Model):
    """
    The `Model` class basically needs a `forward` method to be able to process
    the input. It can do whatever we want, though, as long as `forward` can be
    differentiated.

    We're passing abstract classes as inputs, so that our model can use
    different types of embeddings and encoder. This allows us to replace
    the embedding with ELMo or BERT without changing the model code, for
    example, or replace the LSTM with a GRU or Transformer in the same way.

    Refer to the imports as an explanation to these abstract classes.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        # Our model has different encoders for each of the fields (passage,
        # answer and question). These could theoretically be different for each
        # field, but for now we're using the same. Hence, we clone the provided
        # encoder.
        self.p_encoder, self.q_encoder, self.a_encoder = util.clone(encoder, 3)

        # We're using a hidden layer to build the output from each encoder.
        # As this can't really change, it's not passed as input.
        # The size has to be the size of concatenating the encoder outputs,
        # since that's how we're combining them in the computation. As they're
        # the same, just multiply the first encoder output by 3.
        # The output of the model (which is the output of this layer) has to
        # have size equal to the number of classes.
        hidden_dim = self.p_encoder.get_output_dim() * 3
        self.hidden2logit = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=vocab.get_vocab_size('label')
        )

        # Categorical (as this is a classification task) accuracy
        self.accuracy = CategoricalAccuracy()
        # CrossEntropyLoss is a combinational of LogSoftmax and
        # Negative Log Likelihood. We won't directly use Softmax in training.
        self.loss = torch.nn.CrossEntropyLoss()

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a_mask = util.get_text_field_mask(answer)

        # We create the embeddings from the input text
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a_emb = self.word_embeddings(answer)
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_enc_out = self.p_encoder(p_emb, p_mask)
        q_enc_out = self.p_encoder(q_emb, q_mask)
        a_enc_out = self.p_encoder(a_emb, a_mask)

        # We then concatenate the representations from each encoder
        encoder_out = torch.cat((p_enc_out, q_enc_out, a_enc_out), 1)
        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        logits = self.hidden2logit(encoder_out)
        # We also compute the class with highest likelihood (our prediction)
        prediction = torch.argmax(logits)
        output = {"logits": logits, "class": prediction}

        # Labels are optional. If they're present, we calculate the accuracy
        # and the loss function.
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss(logits, label)

        # The output is the dict we've been building, with the logits, loss
        # and the prediction.
        return output

    # This function computes the metrics we want to see during training.
    # For now, we only have the accuracy metric, but we could have a number
    # of different metrics here.
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
