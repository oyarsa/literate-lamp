from typing import Dict, cast

import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util


class QuestionModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 embedding_dropout: float = 0.5,
                 encoder_dropout: float = 0.5):
        super(QuestionModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout)
        self.encoder_dropout = torch.nn.Dropout(encoder_dropout)

    def get_output_dim(self) -> int:
        return cast(int, self.encoder.get_output_dim())

    @overrides
    def forward(self, question: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = util.get_text_field_mask(question)
        embeddings = self.word_embeddings(question)
        embeddings = self.embedding_dropout(embeddings)

        encoding = self.encoder(embeddings, mask)
        encoding = self.encoder_dropout(encoding)
        return cast(torch.Tensor, encoding)
