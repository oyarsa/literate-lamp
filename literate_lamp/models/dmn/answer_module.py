from typing import Dict, cast

from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder

import torch
from allennlp.nn import util


class AnswerModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 embedding_dropout: float = 0.5,
                 encoder_dropout: float = 0.5):
        super(AnswerModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.encoder_dropout = torch.nn.Dropout(encoder_dropout)

    @overrides
    def forward(self, answer: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = util.get_text_field_mask(answer)
        embeddings = self.word_embeddings(answer)
        embeddings = self.embedding_dropout(embeddings)

        encoding = self.encoder(embeddings, mask)
        encoding = self.encoder_dropout(encoding)
        return cast(torch.Tensor, encoding)
