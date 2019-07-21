"""
Uses the HAN architecture and then combines the output with a relation matrix
obtained from ConceptNet.
"""
from typing import Dict, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules import LayerNorm
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from models.util import seq_over_seq
from layers import LinearAttention, BilinearAttention


class RelationalXL(BaseModel):
    """
    Uses hierarchical RNNs to build a sentence representation (from the
    windows) and then build a document representation from those sentences,
    combining it with commonsense relations.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 text_encoder: Seq2SeqEncoder,
                 relation_encoder: Seq2VecEncoder,
                 vocab: Vocabulary,
                 encoder_dropout: float = 0.5
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.text_encoder = text_encoder
        self.text_attn = LinearAttention(
            input_dim=text_encoder.get_output_dim()
        )

        self.relation_encoder = relation_encoder
        self.relation_attn = BilinearAttention(
            vector_dim=text_encoder.get_output_dim(),
            matrix_dim=relation_encoder.get_output_dim()
        )

        hidden_dim = (text_encoder.get_output_dim() +
                      relation_encoder.get_output_dim())
        self.output = torch.nn.Linear(
            in_features=hidden_dim,
            out_features=1
        )

    def _forward_internal(self,
                          text: Dict[str, torch.Tensor],
                          relations: Dict[str, torch.Tensor]
                          ) -> torch.Tensor:
        t_mask = util.get_text_field_mask(text)
        t_emb = self.word_embeddings(text)

        t_hiddens = self.encoder_dropout(self.text_encoder(t_emb, t_mask))
        t_encoding = self.text_attn(t_hiddens)

        r_masks = util.get_text_field_mask(relations, num_wrapping_dims=1)
        r_embs = self.word_embeddings(relations)

        r_sentence_encodings = self.encoder_dropout(
            seq_over_seq(self.relation_encoder, r_embs, r_masks)
        )
        r_attn = self.relation_attn(
            vector=t_encoding,
            matrix=r_sentence_encodings
        )
        r_encoding = util.weighted_sum(r_sentence_encodings, r_attn)

        final = torch.cat((t_encoding, r_encoding), dim=-1)

        logit = self.output(final).squeeze(-1)
        return logit

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                string0: Dict[str, torch.Tensor],
                string1: Dict[str, torch.Tensor],
                rel0: Dict[str, torch.Tensor],
                rel1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        logit0 = self._forward_internal(string0, rel0)
        logit1 = self._forward_internal(string0, rel1)
        logits = torch.stack((logit0, logit1), dim=-1)

        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(logits, label)

        return output
