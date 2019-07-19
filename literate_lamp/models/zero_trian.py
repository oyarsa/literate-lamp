"""
This is Trian without the overlapping, handcrafted, POS, NER and relation
features. Just text encoding. The version with text + relation is SimpleTrian,
and the full one is Trian.
"""
from typing import Dict, Optional

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn import util

from models.base_model import BaseModel
from layers import LinearSelfAttention, BilinearAttention, SequenceAttention


class ZeroTrian(BaseModel):
    """
    Refer to `BaselineClassifier` for details on how this works.

    This one is based on that, but adds attention layers after RNNs.
    These are:
        - self-attention (LinearAttention, ignoring the second tensor)
        for question and answer
        - a BilinearAttention for attenting a question to a passage

    This means that our RNNs are Seq2Seq now, returning the entire hidden
    state as matrices. For the passage-to-question e need a vector for the
    question state, so we use the hidden state matrix weighted by the
    attention result.

    We use the attention vectors as weights for the RNN hidden states.

    After attending the RNN states, we concat them and feed into a FFN.
    """

    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 p_encoder: Seq2SeqEncoder,
                 q_encoder: Seq2SeqEncoder,
                 a_encoder: Seq2SeqEncoder,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0.0,
                 encoder_dropout: float = 0.0) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.word_embeddings = word_embeddings

        if embedding_dropout > 0:
            self.embedding_dropout = torch.nn.Dropout(p=embedding_dropout)
        else:
            self.embedding_dropout = lambda x: x

        if encoder_dropout > 0:
            self.encoder_dropout = torch.nn.Dropout(p=encoder_dropout)
        else:
            self.encoder_dropout = lambda x: x

        embedding_dim = word_embeddings.get_output_dim()
        self.p_q_match = SequenceAttention(input_dim=embedding_dim)
        self.a_p_match = SequenceAttention(input_dim=embedding_dim)
        self.a_q_match = SequenceAttention(input_dim=embedding_dim)

        # Our model has different encoders for each of the fields (passage,
        # answer and question).
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.a_encoder = a_encoder

        # Attention layers: passage-question, question-self, answer-self
        self.p_q_attn = BilinearAttention(
            vector_dim=self.q_encoder.get_output_dim(),
            matrix_dim=self.p_encoder.get_output_dim(),
        )
        self.q_self_attn = LinearSelfAttention(
            input_dim=self.q_encoder.get_output_dim()
        )
        self.a_self_attn = LinearSelfAttention(
            input_dim=self.a_encoder.get_output_dim()
        )
        self.p_a_bilinear = torch.nn.Linear(
            in_features=self.p_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )
        self.q_a_bilinear = torch.nn.Linear(
            in_features=self.q_encoder.get_output_dim(),
            out_features=self.a_encoder.get_output_dim()
        )

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

        # Every sample in a batch has to have the same size (as it's a tensor),
        # so smaller entries are padded. The mask is used to counteract this
        # padding.
        p_mask = util.get_text_field_mask(passage)
        q_mask = util.get_text_field_mask(question)
        a0_mask = util.get_text_field_mask(answer0)
        a1_mask = util.get_text_field_mask(answer1)

        # We create the embeddings from the input text
        p_emb = self.embedding_dropout(self.word_embeddings(passage))
        q_emb = self.embedding_dropout(self.word_embeddings(question))
        a0_emb = self.embedding_dropout(self.word_embeddings(answer0))
        a1_emb = self.embedding_dropout(self.word_embeddings(answer1))

        # We compute the Sequence Attention
        # First the scores
        p_q_scores = self.p_q_match(p_emb, q_emb, q_mask)
        a0_q_scores = self.a_q_match(a0_emb, q_emb, q_mask)
        a1_q_scores = self.a_q_match(a1_emb, q_emb, q_mask)
        a0_p_scores = self.a_p_match(a0_emb, p_emb, p_mask)
        a1_p_scores = self.a_p_match(a1_emb, p_emb, p_mask)

        p_q_match = p_q_scores.bmm(q_emb)
        a0_q_match = a0_q_scores.bmm(q_emb)
        a1_q_match = a1_q_scores.bmm(q_emb)
        a0_p_match = a0_p_scores.bmm(p_emb)
        a1_p_match = a1_p_scores.bmm(p_emb)

        # We combine the inputs to our encoder
        p_input = torch.cat((p_emb, p_q_match), dim=2)
        a0_input = torch.cat((a0_emb, a0_p_match, a0_q_match), dim=2)
        a1_input = torch.cat((a1_emb, a1_p_match, a1_q_match), dim=2)
        q_input = q_emb

        # Then we use those (along with the masks) as inputs for
        # our encoders
        p_hiddens = self.encoder_dropout(self.p_encoder(p_input, p_mask))
        q_hiddens = self.encoder_dropout(self.q_encoder(q_input, q_mask))
        a0_hiddens = self.encoder_dropout(self.a_encoder(a0_input, a0_mask))
        a1_hiddens = self.encoder_dropout(self.a_encoder(a1_input, a1_mask))

        # We compute the self-attention scores
        q_attn = self.q_self_attn(q_hiddens, q_hiddens, q_mask)
        a0_attn = self.a_self_attn(a0_hiddens, a0_hiddens, a0_mask)
        a1_attn = self.a_self_attn(a1_hiddens, a1_hiddens, a1_mask)

        # Then we weight the hidden-states with those scores
        q_weighted = util.weighted_sum(q_hiddens, q_attn)
        a0_weighted = util.weighted_sum(a0_hiddens, a0_attn)
        a1_weighted = util.weighted_sum(a1_hiddens, a1_attn)

        # We weight the text states with a passage-question bilinear attention
        p_q_attn = self.p_q_attn(q_weighted, p_hiddens, p_mask)
        p_weighted = util.weighted_sum(p_hiddens, p_q_attn)

        # Calculate the outputs for each answer, from passage-answer and
        # question-answer attention again
        out_0 = (self.p_a_bilinear(p_weighted) * a0_weighted).sum(dim=1)
        out_0 += (self.q_a_bilinear(q_weighted) * a0_weighted).sum(dim=1)

        out_1 = (self.p_a_bilinear(p_weighted) * a1_weighted).sum(dim=1)
        out_1 += (self.q_a_bilinear(q_weighted) * a1_weighted).sum(dim=1)

        # Output vector is 2-dim vector with both logits
        logits = torch.stack((out_0, out_1), dim=1)
        # # We softmax to turn those logits into probabilities
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
