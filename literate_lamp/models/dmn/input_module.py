from typing import Dict, cast

import torch
from torch.nn.init import xavier_normal_
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util

from models.util import seq_over_seq, initalise_weights


class InputModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 embedding_dropout: float = 0.5,
                 encoder_dropout: float = 0.5):
        super(InputModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.sentence_encoder = sentence_encoder
        self.document_encoder = document_encoder

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.encoder_dropout = torch.nn.Dropout(encoder_dropout)

        initalise_weights(xavier_normal_, self.sentence_encoder)
        initalise_weights(xavier_normal_, self.document_encoder)

    def get_output_dim(self) -> int:
        return cast(int, self.document_encoder.get_output_dim()) // 2

    @overrides
    def forward(self, sentences: Dict[str, torch.Tensor]) -> torch.Tensor:
        sentences_msks = util.get_text_field_mask(sentences,
                                                  num_wrapping_dims=1)
        sentences_embs = self.word_embeddings(sentences)
        sentences_embs = self.embedding_dropout(sentences_embs)

        sentences_encs = seq_over_seq(self.sentence_encoder, sentences_embs,
                                      sentences_msks)
        sentences_encs = self.encoder_dropout(sentences_encs)

        document_enc = self.document_encoder(sentences_encs, mask=None)
        document_enc = self.encoder_dropout(document_enc)

        mid = document_enc.size(2) // 2
        document_enc = document_enc[:, :, :mid] + document_enc[:, :, mid:]
        return cast(torch.Tensor, document_enc)
