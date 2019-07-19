from typing import Dict, Optional
import copy

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary

from models.base_model import BaseModel
from layers import BilinearMatrixAttention


class Dcmn(BaseModel):
    """
    Uses attention co-matching to improve on improve on BERT's embeddings.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0.0
                 ) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        if embedding_dropout > 0:
            self.emb_dropout = torch.nn.Dropout(p=embedding_dropout)
        else:
            self.emb_dropout = torch.nn.Identity()

        embedding_dim = word_embeddings.get_output_dim()

        self.p_a_match = BilinearMatrixAttention(
            matrix1_dim=embedding_dim,
            matrix2_dim=embedding_dim
        )
        self.p_q_match = BilinearMatrixAttention(
            matrix1_dim=embedding_dim,
            matrix2_dim=embedding_dim
        )

        combination = torch.nn.Sequential(
            torch.nn.Linear(2 * embedding_dim, embedding_dim),
            torch.nn.ReLU(inplace=True)
        )
        self.p_a_combination = copy.deepcopy(combination)
        self.a_combination = copy.deepcopy(combination)
        self.p_q_combination = copy.deepcopy(combination)
        self.q_combination = copy.deepcopy(combination)

        self.output = torch.nn.Linear(4 * embedding_dim, 1)

    def _forward_internal(self,
                          passage: Dict[str, torch.Tensor],
                          question: Dict[str, torch.Tensor],
                          answer: Dict[str, torch.Tensor],
                          ) -> torch.Tensor:
        # H_p
        p_embs = self.emb_dropout(self.word_embeddings(passage))
        # H_q
        q_embs = self.emb_dropout(self.word_embeddings(question))
        # H_a
        a_embs = self.emb_dropout(self.word_embeddings(answer))

        # W
        weights_p_a = self.p_a_match(p_embs, a_embs)
        match_p_a = weights_p_a.bmm(a_embs)
        match_a = weights_p_a.transpose(-2, -1).bmm(p_embs)

        # W'
        weights_p_q = self.p_q_match(p_embs, q_embs)
        match_p_q = weights_p_q.bmm(q_embs)
        match_q = weights_p_q.transpose(-2, -1).bmm(p_embs)

        # S_p
        p_a_cat = torch.cat((match_p_a - p_embs, match_p_a * p_embs), dim=-1)
        combined_p_a = self.p_a_combination(p_a_cat)
        # S_a
        a_cat = torch.cat((match_a - a_embs, match_a * a_embs), dim=-1)
        combined_a = self.a_combination(a_cat)

        # S_p'
        p_q_cat = torch.cat((match_p_q - p_embs, match_p_q * p_embs), dim=-1)
        combined_p_q = self.p_q_combination(p_q_cat)
        # S_a
        q_cat = torch.cat((match_q - q_embs, match_q * q_embs), dim=-1)
        combined_q = self.q_combination(q_cat)

        # Row-wise Max Pooling
        pooled_p_a = combined_p_a.max(dim=1)[0]
        pooled_a = combined_a.max(dim=1)[0]
        pooled_p_q = combined_p_q.max(dim=1)[0]
        pooled_q = combined_q.max(dim=1)[0]

        pooled = torch.cat((pooled_p_a, pooled_a, pooled_p_q, pooled_q),
                           dim=-1)
        logit = self.output(pooled).squeeze(-1)
        return logit

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.

    def forward(self,
                passage_id: Dict[str, torch.Tensor],
                question_id: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        logit0 = self._forward_internal(passage, question, answer0)
        logit1 = self._forward_internal(passage, question, answer1)
        logits = torch.stack((logit0, logit1), dim=-1)

        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(logits, label)

        return output
