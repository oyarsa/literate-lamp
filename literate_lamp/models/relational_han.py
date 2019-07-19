"""
Uses the HAN architecture and then combines the output with a relation matrix
obtained from ConceptNet.
"""
from typing import Dict, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules import LayerNorm
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from models.util import hierarchical_seq_over_seq, seq_over_seq
from layers import LinearAttention


class RelationalHan(BaseModel):
    """
    Uses hierarchical RNNs to build a sentence representation (from the
    windows) and then build a document representation from those sentences,
    combining it with commonsense relations.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 document_encoder: Seq2SeqEncoder,
                 relation_encoder: Seq2SeqEncoder,
                 document_relation_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.5,
                 ffn_dropout: float = 0.2
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.sentence_encoder = sentence_encoder

        self.sentence_attn = LinearAttention(
            input_dim=self.sentence_encoder.get_output_dim()
        )

        self.document_encoder = document_encoder
        self.document_attn = LinearAttention(
            input_dim=self.document_encoder.get_output_dim()
        )

        self.relation_encoder = relation_encoder
        self.relation_attn = LinearAttention(
            input_dim=self.relation_encoder.get_output_dim()
        )

        linear_dim = document_encoder.get_output_dim()
        feedforward_dim = 4 * linear_dim

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(linear_dim, feedforward_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(ffn_dropout),
            torch.nn.Linear(feedforward_dim, linear_dim),
            torch.nn.Dropout(ffn_dropout)
        )
        self.norm = LayerNorm(linear_dim)

        self.output = torch.nn.Linear(
            in_features=linear_dim,
            out_features=1
        )

    def _forward_internal(self,
                          bert: Dict[str, torch.Tensor],
                          answer: Dict[str, torch.Tensor],
                          relations: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        t_masks = util.get_text_field_mask(bert, num_wrapping_dims=1)
        t_embs = self.word_embeddings(bert)

        t_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t_embs, t_masks)
        )
        t_sentence_encodings = seq_over_seq(self.sentence_attn,
                                            t_sentence_hiddens)

        t_document_hiddens = self.encoder_dropout(
            self.document_encoder(t_sentence_encodings, mask=None)
        )
        t_document_encoding = self.document_attn(t_document_hiddens)

        logit = self.ffn(t_document_encoding)
        logit = self.norm(logit + t_document_encoding)
        logit = self.output(t_document_encoding).squeeze(-1)
        return logit

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
                p_a0_rel: Dict[str, torch.Tensor],
                p_a1_rel: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        logit0 = self._forward_internal(bert0, answer0, p_a0_rel)
        logit1 = self._forward_internal(bert1, answer1, p_a1_rel)
        logits = torch.stack((logit0, logit1), dim=-1)

        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(logits, label)

        return output
