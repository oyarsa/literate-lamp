from typing import Dict, cast

import torch
from overrides import overrides
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util


class QuestionModule(torch.nn.Module):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super(QuestionModule, self).__init__()

        self.word_embeddings = word_embeddings
        self.encoder = encoder

    @overrides
    def forward(self, question: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = util.get_text_field_mask(question)
        embeddings = self.word_embeddings(question)

        encoding = self.encoder(embeddings, mask)
        return cast(torch.Tensor, encoding)
