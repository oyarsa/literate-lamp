"Model that uses the RelationalTransformer to encoded Text/Relation."
from typing import Dict, Optional

import torch
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from models.util import hierarchical_seq_over_seq, seq_over_seq
from layers import RelationalTransformerEncoder, LinearAttention


class RelationalTransformerModel(BaseModel):
    "Uses the Relational Transformer to combine Relations and Text."

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2SeqEncoder,
                 # document_encoder: Seq2SeqEncoder,
                 relation_sentence_encoder: Seq2SeqEncoder,
                 relational_encoder: RelationalTransformerEncoder,
                 rel_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 # relation_encoder: Optional[Seq2VecEncoder] = None,
                 encoder_dropout: float = 0.0
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings
        self.rel_embeddings = rel_embeddings
        self.relation_sentence_encoder = relation_sentence_encoder
        # self.relation_encoder = relation_encoder

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        self.sentence_encoder = sentence_encoder

        self.sentence_attn = LinearAttention(
            input_dim=self.sentence_encoder.get_output_dim()
        )
        self.relation_sentence_attn = LinearAttention(
            input_dim=self.sentence_encoder.get_output_dim()
        )

        # self.document_encoder = document_encoder
        self.relational_encoder = relational_encoder
        self.document_attn = LinearAttention(
            input_dim=self.relational_encoder.get_output_dim()
        )

        self.output = torch.nn.Linear(
            in_features=self.relational_encoder.get_output_dim(),
            out_features=1,
            bias=False
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                bert0: Dict[str, torch.Tensor],
                bert1: Dict[str, torch.Tensor],
                p_a0_rel: Dict[str, torch.Tensor],
                p_a1_rel: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        r0_masks = util.get_text_field_mask(p_a0_rel, num_wrapping_dims=1)
        r1_masks = util.get_text_field_mask(p_a1_rel, num_wrapping_dims=1)

        r0_embs = self.rel_embeddings(p_a0_rel)
        r1_embs = self.rel_embeddings(p_a1_rel)

        r0_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.relation_sentence_encoder,
                                      r0_embs, r0_masks))
        r1_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.relation_sentence_encoder, r1_embs,
                                      r1_masks))

        r0_sentence_encodings = seq_over_seq(self.relation_sentence_attn,
                                             r0_sentence_hiddens)
        r1_sentence_encodings = seq_over_seq(self.relation_sentence_attn,
                                             r1_sentence_hiddens)

        # We create the embeddings from the input text
        t0_masks = util.get_text_field_mask(bert0, num_wrapping_dims=1)
        t1_masks = util.get_text_field_mask(bert1, num_wrapping_dims=1)

        t0_embs = self.word_embeddings(bert0)
        t1_embs = self.word_embeddings(bert1)

        t0_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t0_embs,
                                      t0_masks))
        t1_sentence_hiddens = self.encoder_dropout(
            hierarchical_seq_over_seq(self.sentence_encoder, t1_embs,
                                      t1_masks))

        t0_sentence_encodings = seq_over_seq(self.sentence_attn,
                                             t0_sentence_hiddens)
        t1_sentence_encodings = seq_over_seq(self.sentence_attn,
                                             t1_sentence_hiddens)

        # Joining Text and Knowledge
        t0_document_hiddens = self.relational_encoder(
            src=t0_sentence_encodings,
            kb=r0_sentence_encodings,
        )
        t1_document_hiddens = self.relational_encoder(
            src=t1_sentence_encodings,
            kb=r1_sentence_encodings
        )

        t0_document_encoding = self.document_attn(t0_document_hiddens)
        t1_document_encoding = self.document_attn(t1_document_hiddens)

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
