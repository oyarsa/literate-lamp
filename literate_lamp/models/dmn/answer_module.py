from typing import Dict, cast

from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder

import torch
from allennlp.nn import util


class AnswerModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super(AnswerModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.encoder = encoder

    @overrides
    def forward(self, answer: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = util.get_text_field_mask(answer)
        embeddings = self.word_embeddings(answer)

        encoding = self.encoder(embeddings, mask)
        return cast(torch.Tensor, encoding)
