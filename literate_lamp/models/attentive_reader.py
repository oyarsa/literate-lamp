"Implements the AttentiveReader class."
from typing import Dict, Optional

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from layers import BilinearAttention


class AttentiveReader(BaseModel):
    """
    Refer to `BaselineClassifier` for details on how this works.

    This is an implementation of the Attentive Reader, the baselien for the
    SemEval-2018 Task 11.

    It uses a Seq2Seq to encode the passage, and Seq2Vec to encode the question
    and answer. Then it weighs the passage states according to question
    bilinear attention.

    The prediction is another bilinear multiplication of the weighted passage
    and the answer, fed into a sigmoid.

    NOTE: The original had both answers on the input, and the output would
    be a vector of size 2, and the output would be the probability of each
    being right. I can see the theoretical difference, but I'm keeping it
    this way for now, because this is also what other models (TriAN) use.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2VecEncoder,
                 a_encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)
        self.word_embeddings = word_embeddings

        # Our model has different encoders for each of the fields (passage,
        # answer and question).
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.a_encoder = a_encoder

        self.emb_dropout = torch.nn.Dropout(p=0.5)

        # Attention layers: passage-question, passage-answer
        self.p_q_attn = BilinearAttention(
            vector_dim=self.q_encoder.get_output_dim(),
            matrix_dim=self.p_encoder.get_output_dim(),
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )

    # This is the computation bit of the model. The arguments of this function
    # are the fields from the `Instance` we created, as that's what's going to
    # be passed to this. We also have the optional `label`, which is only
    # available at training time, used to calculate the loss.
    def forward(self,
                metadata: Dict[str, torch.Tensor],
                passage: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.emb_dropout(self.word_embeddings(passage))
        q_emb = self.emb_dropout(self.word_embeddings(question))
        a0_emb = self.emb_dropout(self.word_embeddings(answer0))
        a1_emb = self.emb_dropout(self.word_embeddings(answer1))
        # Then we use those embeddings (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.p_encoder(p_emb, p_mask)
        q_hidden = self.q_encoder(q_emb, q_mask)
        a0_hidden = self.a_encoder(a0_emb, a0_mask)
        a1_hidden = self.a_encoder(a1_emb, a1_mask)

        # We weight the text hidden states according to text-question attention
        p_q_attn = self.p_q_attn(q_hidden, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        # We combine the output with a bilinear attention from text to answer
        out0 = (self.p_a_bilinear(p_weighted) * a0_hidden).sum(dim=1)
        out1 = (self.p_a_bilinear(p_weighted) * a1_hidden).sum(dim=1)
        logits = torch.stack((out0, out1), dim=1)
        # We also compute the class with highest likelihood (our prediction)
        prob = torch.softmax(logits, dim=-1)
        output = {"logits": logits, "prob": prob}

        # Labels are optional. If they're present, we calculate the accuracy
        # and the loss function.
        if label is not None:
            self.accuracy(prob, label)
            output["loss"] = self.loss(prob, label)

        # The output is the dict we've been building, with the logits, loss
        # and the prediction.
        return output
