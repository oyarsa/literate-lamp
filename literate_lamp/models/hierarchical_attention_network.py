"""
Hierarchical Attention Networks for Document Classification
https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
"""
from typing import Dict, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from models.util import hierarchical_seq_over_seq
from layers import LinearSelfAttention


class HierarchicalAttentionNetwork(BaseModel):
    """
    Uses hierarchical RNNs to build a sentence representation (from the
    windows) and then build a document representation from those sentences.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 document_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.0
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.sentence_encoder = sentence_encoder

        self.sentence_attn = LinearSelfAttention(
            input_dim=self.sentence_encoder.get_output_dim(),
            bias=True
        )

        self.document_encoder = document_encoder
        self.document_attn = LinearSelfAttention(
            input_dim=self.document_encoder.get_output_dim(),
            bias=True
        )

        self.output = torch.nn.Linear(
            in_features=document_encoder.get_output_dim(),
            out_features=1
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                bert0: Dict[str, torch.Tensor],
                bert1: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        t0_masks = util.get_text_field_mask(bert0, num_wrapping_dims=1)
        t1_masks = util.get_text_field_mask(bert1, num_wrapping_dims=1)

        # We create the embeddings from the input text
        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t0_embs,
                                      t0_masks))
        t1_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t1_embs,
                                      t1_masks))

        t0_sentence_attns = self.sentence_attn(
            t0_sentence_hiddens, t0_sentence_hiddens)
        t1_sentence_attns = self.sentence_attn(
            t1_sentence_hiddens, t1_sentence_hiddens)

        t0_sentence_encodings = util.weighted_sum(
            t0_sentence_hiddens, t0_sentence_attns)
        t1_sentence_encodings = util.weighted_sum(
            t1_sentence_hiddens, t1_sentence_attns)

        t0_document_hiddens = self.encoder_dropout(self.document_encoder(
            t0_sentence_encodings, mask=None))
        t1_document_hiddens = self.encoder_dropout(self.document_encoder(
            t1_sentence_encodings, mask=None))

        t0_document_attn = self.document_attn(
            t0_document_hiddens, t0_document_hiddens)
        t1_document_attn = self.document_attn(
            t1_document_hiddens, t1_document_hiddens)

        t0_document_encoding = util.weighted_sum(
            t0_document_hiddens, t0_document_attn)
        t1_document_encoding = util.weighted_sum(
            t1_document_hiddens, t1_document_attn)

        t0_final = t0_document_encoding
        t1_final = t1_document_encoding

        # Joining everything and getting the result
        logit0 = self.output(t0_final).squeeze(-1)
        logit1 = self.output(t1_final).squeeze(-1)

        logits = torch.stack((logit0, logit1), dim=-1)
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
