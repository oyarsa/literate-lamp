"""
Implementation of a Dynamic Memory Network.
"""
from typing import Dict, Optional

import torch
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.data.vocabulary import Vocabulary

from models.base_model import BaseModel
from models.dmn.input_module import InputModule
from models.dmn.answer_module import AnswerModule
from models.dmn.question_module import QuestionModule
from models.dmn.output_module import OutputModule
from models.dmn.memory_module import MemoryModule


class Dmn(BaseModel):
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
                 sentence_encoder: Seq2VecEncoder,
                 document_encoder: Seq2VecEncoder,
                 question_encoder: Seq2VecEncoder,
                 answer_encoder: Seq2VecEncoder,
                 passes: int,
                 vocab: Vocabulary,
                 embedding_dropout: float = 0.0,
                 encoder_dropout: float = 0.5) -> None:
        # We have to pass the vocabulary to the constructor.
        super().__init__(vocab)

        self.hidden_dim = sentence_encoder.get_output_dim()
        self.passes = passes

        self.input_module = InputModule(
            word_embeddings=word_embeddings,
            sentence_encoder=sentence_encoder,
            document_encoder=document_encoder,
            embedding_dropout=embedding_dropout,
            encoder_dropout=encoder_dropout
        )
        self.question_module = QuestionModule(
            word_embeddings=word_embeddings,
            encoder=question_encoder,
            embedding_dropout=embedding_dropout,
            encoder_dropout=encoder_dropout
        )
        self.answer_module = AnswerModule(
            word_embeddings=word_embeddings,
            encoder=answer_encoder,
            embedding_dropout=embedding_dropout,
            encoder_dropout=encoder_dropout
        )
        self.memory_module = MemoryModule(
            hidden_dim=self.input_module.get_output_dim(),
            num_hops=passes,
            dropout=encoder_dropout
        )
        self.output_module = OutputModule(
            memory_size=self.memory_module.get_output_dim(),
            answer_size=self.answer_module.get_output_dim(),
            num_labels=1
        )

        _assert_equal(self.input_module, self.question_module)
        _assert_equal(self.memory_module, self.question_module)

    def forward(self,
                metadata: Dict[str, torch.Tensor],
                sentences: Dict[str, torch.Tensor],
                question: Dict[str, torch.Tensor],
                answer0: Dict[str, torch.Tensor],
                answer1: Dict[str, torch.Tensor],
                label: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        facts_encoded = self.input_module(sentences)
        question_encoded = self.question_module(question)
        answer0_encoded = self.answer_module(answer0)
        answer1_encoded = self.answer_module(answer1)
        memory0 = question_encoded
        memory1 = question_encoded

        for hop in range(self.passes):
            memory0 = self.memory_module(facts_encoded, question_encoded,
                                         answer0_encoded, memory0, hop)
            memory1 = self.memory_module(facts_encoded, question_encoded,
                                         answer1_encoded, memory1, hop)

        out_0 = self.output_module(memory0, answer0_encoded)
        out_1 = self.output_module(memory1, answer1_encoded)

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


def _assert_equal(a: torch.nn.Module, b: torch.nn.Module) -> None:
    name_a = type(a).__name__
    a_size = a.get_output_dim()
    b_size = b.get_output_dim()
    name_b = type(b).__name__
    error_str = f'Output size of {name_a} and {name_b} shoud match, but '\
        f'{a_size} != {b_size}'
    assert a_size == b_size, error_str
