from typing import Dict, cast

import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util

from models.util import seq_over_seq


class InputModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 dropout: float = 0.1):
        super(InputModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.sentence_encoder = sentence_encoder
        self.document_encoder = document_encoder

        self.dropout = torch.nn.Dropout(dropout)

    @overrides
    def forward(self, sentences: Dict[str, torch.Tensor]) -> torch.Tensor:
        sentences_msks = util.get_text_field_mask(sentences,
                                                  num_wrapping_dims=1)
        sentences_embs = self.word_embeddings(sentences)

        sentences_encs = seq_over_seq(self.sentence_encoder, sentences_embs,
                                      sentences_msks)
        sentences_encs = self.dropout(sentences_encs)

        document_enc = self.document_encoder(sentences_encs, mask=None)
        return cast(torch.Tensor, document_enc)
