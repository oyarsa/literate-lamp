"Baseline method. No commonsense, no sophisticated architecture."
from typing import Dict, Optional

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary

from models.base_model import BaseModel
import util


class BaselineClassifier(BaseModel):
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
        hidden_dim = self.p_encoder.get_output_dim() * 4
        self.hidden2logit = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=vocab.get_vocab_size('label')
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.word_embeddings(passage)
        q_emb = self.word_embeddings(question)
        a0_emb = self.word_embeddings(answer0)
        a1_emb = self.word_embeddings(answer1)
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_enc_out = self.p_encoder(p_emb, p_mask)
        q_enc_out = self.q_encoder(q_emb, q_mask)
        a0_enc_out = self.a_encoder(a0_emb, a0_mask)
        a1_enc_out = self.a_encoder(a1_emb, a1_mask)

        # We then concatenate the representations from each encoder
        encoder_out = torch.cat(
            (p_enc_out, q_enc_out, a0_enc_out, a1_enc_out), 1)
        # Finally, we pass each encoded output tensor to the feedforward layer
        # to produce logits corresponding to each class.
        logits = self.hidden2logit(encoder_out)
        # We also compute the class with highest likelihood (our prediction)
        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        # Labels are optional. If they're present, we calculate the accuracy
        # and the loss function.
        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(logits, label)

        # The output is the dict we've been building, with the logits, loss
        # and the prediction.
        return output
